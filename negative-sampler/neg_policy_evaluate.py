
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys

from utils import cuda_

sys.path.append('/home/mengyuan/KGenSam/KG')
from knowledge_graph import global_kg

def get_score(model,s, t):
    # u_e, i_e,_ = torch.split(model.ui_emb, [n_users, n_items,1])
    u_e_idx=Variable(torch.LongTensor([i for i in range(global_kg.user_range[0],global_kg.user_range[1])]))
    i_e_idx=Variable(torch.LongTensor([i for i in range(global_kg.item_range[0]-global_kg.user_range[1],global_kg.item_range[1]-global_kg.user_range[1])]))
    u_e=model.ui_emb(cuda_(u_e_idx))
    i_e=model.ui_emb(cuda_(i_e_idx))

    u_e = u_e[s:t, :]

    score_matrix = torch.matmul(u_e, i_e.t())
    # for u in range(s, t):
    #     pos = global_kg.train_user_dict[u]
    #     idx = pos.index(-1) if -1 in pos else len(pos)
    #     if idx>=global_kg.n_users:
    #         idx=idx-global_kg.n_users
    #     else:
    #         # print('idx:{}'.format(idx))
    #         print('score_matrix:{}'.format(score_matrix.size()))
    #         # print(score_matrix)
    #     print('idx:{} ; u:{} ; s:{} ; pos[:idx]:{} '.format(idx,u,s,pos[:idx]))
    #     score_matrix[u-s][pos[:idx]] = -1e5

    return score_matrix


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


def valid_score(model, ks, n_batchs=4):
    ks = eval(ks)
    train_user_dict, test_user_dict = global_kg.train_user_dict, global_kg.test_user_dict

    n_test_users = len(test_user_dict)

    n_k = len(ks)
    result = {
        "precision": np.zeros(n_k),
        "recall": np.zeros(n_k),
        "ndcg": np.zeros(n_k),
        "hit_ratio": np.zeros(n_k),
    }

    # n_users = model.n_users
    batch_size = n_test_users // n_batchs
    for batch_id in tqdm(range(n_batchs), ascii=True, desc="Evaluate"):
        s = batch_size * batch_id
        t = batch_size * (batch_id + 1)
        if t > n_test_users:
            t = n_test_users
        if s >= t:
            break

        score_matrix = get_score(model, s, t)
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

    return result
