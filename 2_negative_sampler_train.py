
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
sys.path.append('/home/mengyuan/KGenSam')
from utils import cuda_,set_random_seed
set_random_seed()

import warnings
warnings.filterwarnings('ignore')


import torch


import random

import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from prettytable import PrettyTable

sys.path.append('/home/mengyuan/KGenSam/data-helper')
from data_loader import KG_Data_loader
from data_in import load_pretrain_fm_model,load_ns_model
from data_out import save_negative_sampler_model,save_final_negative_sampler_model

sys.path.append('/home/mengyuan/KGenSam/negative-sampler')
from neg_policy_evaluate import valid_score
from neg_policy import KGPolicy


sys.path.append('/home/mengyuan/KGenSam/KG')
from knowledge_graph import global_kg

sys.path.append('/home/mengyuan/KGenSam/FM')
from factorization_machine import FM

sys.path.append('/home/mengyuan/KGenSam/configuration')
from base_config import bcfg

from utils import cuda_,early_stopping, print_dict

def train_one_epoch(args_config,
    recommender,
    sampler,
    train_loader,
    recommender_optim,
    sampler_optim,
    adj_matrix,
    edge_matrix,
    train_data,
    cur_epoch,
    avg_reward,
):

    loss, base_loss, reg_loss = 0, 0, 0
    epoch_reward = 0

    """Train one epoch"""
    num_batch = len(train_loader)
    nb=0
    for batch_data in train_loader:
        nb+=1

        if torch.cuda.is_available():
            batch_data = {k: v.cuda(non_blocking=True) for k, v in batch_data.items()}

        """Train recommender using negtive item provided by negative-sampler"""
        recommender_optim.zero_grad()

        neg = batch_data["neg_i_id"]
        pos = batch_data["pos_i_id"]
        users = batch_data["u_id"]

        selected_neg_items_list, _ = sampler(batch_data, adj_matrix, edge_matrix)
        selected_neg_items = selected_neg_items_list[-1, :]

        train_set = train_data[users]
        in_train = torch.sum(
            selected_neg_items.unsqueeze(1) == train_set.long(), dim=1
        ).byte()
        selected_neg_items[in_train] = neg[in_train]

        # print('selected_neg_items.shape:{}'.format(selected_neg_items.shape))
        # print('selected_neg_items:{}'.format(selected_neg_items))
        base_loss_batch, reg_loss_batch = recommender.get_loss(users, pos, selected_neg_items)
        loss_batch = base_loss_batch + reg_loss_batch

        loss_batch.backward()
        recommender_optim.step()

        if nb%200==0:
            print('batch process {}/{} '.format(nb,num_batch))
            print('loss_batch:{} , bpr_loss:{} , reg_loss:{} '.format(loss_batch,base_loss_batch,reg_loss_batch))

        """Train negative-sampler network"""
        sampler_optim.zero_grad()
        selected_neg_items_list, selected_neg_prob_list = sampler(
            batch_data, adj_matrix, edge_matrix
        )

        with torch.no_grad():
            reward_batch = recommender.get_reward(users, pos, selected_neg_items_list,args_config.k_step)

        epoch_reward += torch.sum(reward_batch)
        reward_batch -= avg_reward

        batch_size = reward_batch.size(1)
        n = reward_batch.size(0) - 1
        R = torch.zeros(batch_size, device=reward_batch.device)
        reward = torch.zeros(reward_batch.size(), device=reward_batch.device)

        gamma = args_config.gamma

        for i, r in enumerate(reward_batch.flip(0)):
            R = r + gamma * R
            reward[n - i] = R

        # print('reward_batch.shape:{};selected_neg_prob_list.shape:{}'.format(reward_batch.shape,selected_neg_prob_list.shape))
        reinforce_loss = -1 * torch.sum(reward_batch * selected_neg_prob_list[0,...])
        if args_config.k_step ==2:
            reinforce_loss = reinforce_loss +(-1 * torch.sum(reward_batch * selected_neg_prob_list[1,...]))

        reinforce_loss.backward()
        sampler_optim.step()

        """record loss in an epoch"""
        loss += loss_batch
        reg_loss += reg_loss_batch
        base_loss += base_loss_batch

    avg_reward = epoch_reward / num_batch
    train_res = PrettyTable()
    train_res.field_names = ["Epoch", "Loss", "BPR-Loss", "Regulation", "AVG-Reward"]
    train_res.add_row(
        [cur_epoch, loss.item(), base_loss.item(), reg_loss.item(), avg_reward.item()]
    )
    print(train_res)

    return loss, base_loss, reg_loss, avg_reward


def build_sampler_graph(n_nodes, edge_threshold, graph):

    adj_matrix = torch.zeros(n_nodes, edge_threshold * 2)
    edge_matrix = torch.zeros(n_nodes, edge_threshold)

    """sample neighbors for each node"""
    for node in tqdm(graph.nodes, ascii=True, desc="Build negative-sampler matrix"):
        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= edge_threshold:
            sampled_edge = random.sample(neighbors, edge_threshold)
            edges = deepcopy(sampled_edge)
        else:
            neg_id = random.sample(
                range(global_kg.item_range[0], global_kg.item_range[1] + 1),
                edge_threshold - len(neighbors),
            )
            node_id = [node] * (edge_threshold - len(neighbors))
            sampled_edge = neighbors + neg_id
            edges = neighbors + node_id

        """concatenate sampled edge with random edge"""
        sampled_edge += random.sample(
            range(global_kg.item_range[0], global_kg.item_range[1] + 1), edge_threshold
        )

        adj_matrix[node] = torch.tensor(sampled_edge, dtype=torch.long)
        edge_matrix[node] = torch.tensor(edges, dtype=torch.long)

    if torch.cuda.is_available():
        adj_matrix = adj_matrix.cuda().long()
        edge_matrix = edge_matrix.cuda().long()

    return adj_matrix, edge_matrix


def build_train_data(train_mat):
    num_user = max(train_mat.keys()) + 1
    num_true = max([len(i) for i in train_mat.values()])

    train_data = torch.zeros(num_user, num_true)

    for i in train_mat.keys():
        true_list = train_mat[i]
        true_list += [-1] * (num_true - len(true_list))
        train_data[i] = torch.tensor(true_list, dtype=torch.long)

    return train_data


def train(train_loader, test_loader, graph,args_config,data_config):
    """build padded training set"""
    train_mat = graph.train_user_dict
    train_data = build_train_data(train_mat)

    recommender=FM(args_config=bcfg.get_FM_parser(),cfg=bcfg)
    if args_config.pretrain_fm:
        # pretrain_r defaut=true
        #一般都是在预训练过的fm上训练neg_sampler
        model_dict=load_pretrain_fm_model()
        recommender.load_state_dict(model_dict)
    cuda_(recommender)
    sampler = KGPolicy(recommender, args_config,data_config)

    ####for break point
    start_epoch=args_config.pretrain_ns_epoch
    if args_config.pretrain_ns_epoch: #不为零
        print("\nnegative-sampler break in epoch {}, recover now!!!!!".format(start_epoch))
        model_dict = load_ns_model(epoch=start_epoch)
        sampler.load_state_dict(model_dict)
    cuda_(sampler)
    train_data = cuda_(train_data.long())

    print("\nSet negative-sampler as: {}".format(str(sampler)))
    print("Set recommender as: {}\n".format(str(recommender)))

    # recommender_optimizer = torch.optim.Adam(recommender.parameters(), lr=args_config.rlr)
    param_bias_ui = list()
    i = 0
    for name, param in recommender.named_parameters():
        print(name, param)
        if i in [0,1]:
            param_bias_ui.append(param)  # bias层
        i += 1
    recommender_optimizer = torch.optim.Adam(param_bias_ui, lr=args_config.rlr)
    sampler_optimizer = torch.optim.Adam(sampler.parameters(), lr=args_config.slr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step, cur_best_pre_0, avg_reward = 0, 0.0, 0

    t0 = time()

    # model_save_filename='Negative_Sampler_model-adj_epoch-{}'.format(args_config.adj_epoch)
    model_save_filename='Negative_Sampler_model-k_step-{}'.format(args_config.k_step)
    # model_save_filename='Negative_Sampler_model'

    for epoch in range(start_epoch,args_config.epoch):
        print('##############################################################')
        print("Epoch {}/{} ".format(epoch,args_config.epoch))
        if epoch==start_epoch or epoch % args_config.adj_epoch == 0:
            """sample adjacency matrix"""
            start = time()
            adj_matrix, edge_matrix = build_sampler_graph(
                graph.entity_range[1] + 1, args_config.edge_threshold, graph.ckg_graph
            )
            print('build_sampler_graph per {} epoch takes: {} secs'.format(args_config.adj_epoch,time() - start))


        loss, base_loss, reg_loss, avg_reward = train_one_epoch(args_config,
                recommender,
                sampler,
                train_loader,
                recommender_optimizer,
                sampler_optimizer,
                adj_matrix,
                edge_matrix,
                train_data,
                epoch,
                avg_reward,
            )

        """VALIDATE"""
        if epoch % args_config.show_step == 0:
            with torch.no_grad():
                ret = valid_score(recommender, args_config.Ks)

            print('-----------evaluate result--------------')
            loss_loger.append(loss)
            rec_loger.append(ret["recall"])
            pre_loger.append(ret["precision"])
            ndcg_loger.append(ret["ndcg"])
            hit_loger.append(ret["hit_ratio"])
            print_dict(ret)
            print('----------------------------------------')

            cur_best_pre_0, stopping_step, should_stop = early_stopping(
                ret["recall"][0],
                cur_best_pre_0,
                stopping_step,
                expected_order="acc",
                flag_step=args_config.flag_step,
            )

            if should_stop:
                print('early stop!!!')
                break

        if epoch%20==0 and epoch!=args_config.epoch-1:
            save_negative_sampler_model(sampler,epoch,filename=model_save_filename)

    save_final_negative_sampler_model(sampler,filename=model_save_filename)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = (
        "Best Iter=[%d]@[%.1f]\n recall=[%s] \n precision=[%s] \n hit=[%s] \n ndcg=[%s]"
        % (
            idx,
            time() - t0,
            "\t".join(["%.5f" % r for r in recs[idx]]),
            "\t".join(["%.5f" % r for r in pres[idx]]),
            "\t".join(["%.5f" % r for r in hit[idx]]),
            "\t".join(["%.5f" % r for r in ndcgs[idx]]),
        )
    )
    print(final_perf)


if __name__ == "__main__":

    """initialize dataset"""
    # global_kg = global_kg

    """initialize args"""
    A = bcfg.get_Negative_Sampler_parser()

    data_config = {
        "n_users": global_kg.n_users,
        "n_items": global_kg.n_items,
        "n_relations": global_kg.n_relations + 2,
        "n_entities": global_kg.n_entities,
        "n_nodes": global_kg.entity_range[1] + 1,
        "item_range": global_kg.item_range,
    }

    # """fix the random seed"""
    # A.seed = 2021
    # random.seed(A.seed)
    # np.random.seed(A.seed)
    # torch.manual_seed(A.seed)
    # torch.cuda.manual_seed(A.seed)

    # yelp 数据过大 batchsize 设大
    if bcfg.data_name=='yelp':
        # A.batch_size*=2
        # A.test_batch_size*=2
        A.num_threads=2
        A.adj_epoch*=10
    print('set data_loader num_threads={}'.format(A.num_threads))

    print("copying global_kg graph for data_loader.. it might take a few minutes")
    graph = deepcopy(global_kg)
    train_loader, test_loader = KG_Data_loader(args_config=A, graph=graph)

    train(
        train_loader=train_loader,
        test_loader=test_loader,
        graph=global_kg,
        args_config=A,
        data_config=data_config
    )
