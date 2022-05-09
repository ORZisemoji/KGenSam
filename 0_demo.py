import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch

import numpy as np

from torch.autograd import Variable

import sys
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
import argparse

parser = argparse.ArgumentParser(description="Run FM")
parser.add_argument('-lr', default=0.02, type=float, dest='lr', help='lr')
parser.add_argument('-flr', default=0.0001, type=float, dest='flr', help='flr')
# means the learning rate of feature similarity learning
parser.add_argument('-reg', default=0.001, type=float, dest='reg', help='reg')
# regularization
parser.add_argument('-decay', default=0.0, type=float, dest='decay', help='decay')
# weight decay
parser.add_argument('-bs', default=64, type=int, dest='bs', help='bs')
# batch size
parser.add_argument('-emb_size', default=64, type=int, dest='emb_size', help='emb_size')
# hidden size/
parser.add_argument('-ip', default=0.01, type=float, dest='ip', help='ip')
# init parameter for hidden
parser.add_argument('-dr', default=0.5, type=float, dest='dr', help='dr')
# dropout ratio
parser.add_argument('-optim', default='Ada', type=str, dest='optim', help='optim')
# optimizer
parser.add_argument('-observe', default=25, type=int, dest='observe', help='observe')
# the frequency of doing evaluation
parser.add_argument('-pretrain_epoch', default=0, type=int, dest='pretrain_epoch', help='pretrain_epoch')
# does it need to load pretrain model
parser.add_argument('-max_epoch', default=250, type=int, dest='max_epoch', help='max_epoch')
# does it need to load pretrain model
parser.add_argument('-updatefeatureemb', default=1, type=int, dest='updatefeatureemb', help='updatefeatureemb')
# 0:不更新属性特征；1：更新
parser.add_argument('-updateuseremb', default=1, type=int, dest='updateuseremb', help='updateuseremb')
# 0:不更新用户特征；1：更新
# parser.add_argument('-command', default=8,type=int, dest='command', help='command')
# # command = 6: normal FM
# # command = 8: with our second type of negative sample
parser.add_argument('-seed', type=int, default=2021, dest='seed', help='seed')
# random seed
A= parser.parse_args(args=[])




recommender=FM(args_config=A,cfg=bcfg)
model_dict=load_pretrain_fm_model()
recommender.load_state_dict(model_dict)

recommender=recommender.cuda()


# ret = valid_score(recommender, args_config.Ks)
n_batchs=4
ks = [20, 40, 60, 80, 100]
train_user_dict, test_user_dict = global_kg.train_user_dict, global_kg.test_user_dict

n_test_users = len(test_user_dict)

n_k = len(ks)
result = {
    "precision": np.zeros(n_k),
    "recall": np.zeros(n_k),
    "ndcg": np.zeros(n_k),
    "hit_ratio": np.zeros(n_k),
}



def cal_ndcg(topk, test_set, num_pos, k):
    n = min(num_pos, k)
    nrange = np.arange(n) + 2
    idcg = np.sum(1 / np.log2(nrange))

    dcg = 0
    for i, s in enumerate(topk):
        if s in test_set:
            dcg += 1 / np.log2(i + 2)

    ndcg = dcg / idcg

    return ndcg

# n_users = recommender.n_users
batch_size = n_test_users // n_batchs
for batch_id in range(n_batchs):
    s = batch_size * batch_id
    t = batch_size * (batch_id + 1)
    if t > n_test_users:
        t = n_test_users
    if s >= t:
        break

    # score_matrix = get_score(recommender, s, t)
    u_e_idx=Variable(torch.LongTensor(global_kg.exist_users))
    i_e_idx=Variable(torch.LongTensor(global_kg.exist_items))
    u_e=recommender.ui_emb(u_e_idx.cuda())
    i_e=recommender.ui_emb(i_e_idx.cuda())

    u_e = u_e[s:t, :]

    score_matrix = torch.matmul(u_e, i_e.t())

for i, k in enumerate(ks):
    precision, recall, ndcg, hr = 0, 0, 0, 0
    _, topk_index = torch.topk(score_matrix, k)
    topk_index = topk_index.cpu().numpy() + global_kg.n_users

    for u in range(s, t):
        gt_pos = test_user_dict[u]
        topk = topk_index[u - s]
        num_pos = len(gt_pos)

        topk_set = set(topk)
        test_set = set(gt_pos)
        num_hit = len(topk_set & test_set)

        precision += num_hit / k
        recall += num_hit / num_pos
        hr += 1 if num_hit > 0 else 0

        ndcg += cal_ndcg(topk, test_set, num_pos, k)

    result["precision"][i] += precision / n_test_users
    result["recall"][i] += recall / n_test_users
    result["ndcg"][i] += ndcg / n_test_users
    result["hit_ratio"][i] += hr / n_test_users

    # max_ind_list=[1]
    # reachable_feature=[i for i in range(5)]
    # max_fea_id = [reachable_feature[i] for i in max_ind_list]
    # [reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
    # [reachable_feature.insert(0, v) for v in max_fea_id[::-1]]






import pickle
import json
import os
fileAndDirString = os.popen("ls -l ./data/").read()
aumdir='/home/mengyuan/KGenSam/data/lastfm/raw-data'
with open('{}/item_dict.pickle'.format(aumdir), 'rb') as f:
    item_dict = pickle.load(f)


featurelist=[]
for k,v in item_dict.items():
    if 'feature_index' in v.keys():
        featurelist.extend(v['feature_index'])
    featurelist=list(set(featurelist))

#
# featurelist=[]
# bigfeaturelist=[]
# for k,v in item_dict.items():
#     if 'feature_index' in v.keys():
#         featurelist.extend(v['feature_index'])
#         bigfeaturelist.extend(v['big_feature_index'])
#     featurelist=list(set(featurelist))
#     bigfeaturelist=list(set(bigfeaturelist))

with open('{}/review_dict_train.json'.format(aumdir), 'rb') as f:
    _train_user_to_items = json.load(f)
with open('{}/review_dict_valid.json'.format(aumdir), 'rb') as f:
    _valid_user_to_items = json.load(f)
with open('{}/review_dict_test.json'.format(aumdir), 'rb') as f:
    _test_user_to_items = json.load(f)

scprdir='/home/mengyuan/crs/data/lastfm/UI_Interaction_data'
with open('{}/review_dict_train.json'.format(scprdir), 'rb') as f:
    _train_user_to_items_scpr = json.load(f)
with open('{}/review_dict_valid.json'.format(scprdir), 'rb') as f:
    _valid_user_to_items_scpr = json.load(f)
with open('{}/review_dict_test.json'.format(scprdir), 'rb') as f:
    _test_user_to_items_scpr = json.load(f)


scprdir='/home/mengyuan/crs/data/lastfm/FM_sample_data'