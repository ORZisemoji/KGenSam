
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
sys.path.append('/home/mengyuan/KGenSam')
from utils import cuda_,set_random_seed

set_random_seed(2021)

import warnings
warnings.filterwarnings('ignore')


import torch


import random
import pickle

import numpy as np

from tqdm import tqdm
from copy import deepcopy


sys.path.append('/home/mengyuan/KGenSam/data-helper')
from data_loader import KG_Data_loader
from data_in import load_pretrain_fm_model,load_as_model
from data_out import save_active_sampler_model,save_final_active_sampler_model

sys.path.append('/home/mengyuan/KGenSam/active-sampler')
from al_policy import ALPolicy,ALagent
from al_policy_evaluate import mean_std

sys.path.append('/home/mengyuan/KGenSam/KG')
from knowledge_graph import global_kg

sys.path.append('/home/mengyuan/KGenSam/FM')
from factorization_machine import FM

sys.path.append('/home/mengyuan/KGenSam/configuration')
from base_config import bcfg

from utils import cuda_




def build_sampler_graph(n_nodes, edge_threshold, graph):
    adj_matrix = torch.zeros(n_nodes, edge_threshold * 2)
    edge_matrix = torch.zeros(n_nodes, edge_threshold)

    """sample neighbors for each node"""
    for node in tqdm(graph.nodes, ascii=True, desc="Build active-sampler matrix"):
        neighbors = list(graph.neighbors(node))
        # if len(neighbors) >= edge_threshold:
        #     sampled_edge = random.sample(neighbors, edge_threshold)
        #     edges = deepcopy(sampled_edge)
        # else:
        #     neg_id = random.sample(
        #         range(global_kg.item_range[0], global_kg.item_range[1] + 1),
        #         edge_threshold - len(neighbors),
        #     )
        #     node_id = [node] * (edge_threshold - len(neighbors))
        #     sampled_edge = neighbors + neg_id
        #     edges = neighbors + node_id
        sampled_edge = neighbors
        edges = deepcopy(sampled_edge)
        # """concatenate sampled edge with random edge"""
        # sampled_edge += random.sample(
        #     range(global_kg.item_range[0], global_kg.item_range[1] + 1), edge_threshold
        # )

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


def train(graph,args_config):
    """build padded training set"""
    train_mat = graph.train_user_dict
    train_data = build_train_data(train_mat)


    sampler = ALPolicy(args_config)
    ####for break point
    start_epoch = args_config.pretrain_as_epoch
    if args_config.pretrain_as_epoch:  # 不为零
        print("\nactive-sampler break in epoch {}, recover now!!!!!".format(start_epoch))
        model_dict = load_as_model(epoch=start_epoch)
        sampler.load_state_dict(model_dict)
    cuda_(sampler)
    print("\nSet active-sampler as: {}".format(str(sampler)))


    sampler_optimizer = torch.optim.Adam(sampler.parameters(), lr=args_config.sllr)


    for epoch in range(start_epoch, args_config.epoch):
        print('##############################################################')
        print("Epoch {}/{} ".format(epoch,args_config.epoch))
        u, item = bcfg.train_list[epoch]
        user_id = int(u)
        item_id = int(item)
        preference_list = bcfg.item_dict[str(item_id)]['feature_index']

        the_agent = ALagent(user_id,item_id,preference_list,sampler, sampler_optimizer, graph)


        """Train one epoch"""
        shapedrewards, logp_actions, p_actions = the_agent.playOneEpisode(epoch)
        loss = the_agent.finishEpisode(shapedrewards, logp_actions, p_actions)

        print("\nActive Sampler Agent loss : {}\n".format(loss))

        sampler=deepcopy(the_agent.policy)
        if epoch%1000==0 and epoch!=args_config.epoch-1:
            save_active_sampler_model(the_agent.policy,epoch)

        """VALIDATE"""
        if epoch % args_config.show_step == 0:
            with torch.no_grad():
                score, bia = mean_std(the_agent.cur_rewards)
            print('-----------epoch {} evaluate result--------------'.format(epoch))
            print('score : {} , bia : {} '.format(score, bia))
            print('-------------------------------------------------')

        """FOR TEST"""
        if epoch==args_config.epoch-1:
            the_evaluate_agent=ALagent(user_id,item_id,preference_list,sampler, sampler_optimizer, graph,test=True)

    save_final_active_sampler_model(sampler)

    """TEST"""
    the_evaluate_agent.test=True
    finalrewards = the_evaluate_agent.get_reward(args_config.mt)
    # print('finalrewards:{}'.format(finalrewards))
    score, bia = mean_std(finalrewards)
    print('---------------------FINAL test result---------------------')
    print('score : {} , bia : {} '.format(score, bia))
    print('-----------------------------------------------------------')



if __name__ == "__main__":
    """initialize dataset"""
    # global_kg = global_kg

    """initialize args"""
    A = bcfg.get_Active_Sampler_parser()

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


    train(
        graph=global_kg,
        args_config=A
    )
