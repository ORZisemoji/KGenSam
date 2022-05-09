#!/usr/local/bin/bash
import os
import re
import sys
import torch.nn.functional as F
import torch
import random
import numpy as np

'''else func'''

def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var

def set_random_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print('SET SEED = {} '.format(seed))

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__




#
# def txt2list(file_src):
#     orig_file = open(file_src, "r")
#     lines = orig_file.readlines()
#     return lines
#
#
# def ensure_dir(dir_path):
#     d = os.path.dirname(dir_path)
#     if not os.path.exists(d):
#         os.makedirs(d)
#
#
# def uni2str(unicode_str):
#     return str(unicode_str.encode("ascii", "ignore")).replace("\n", "").strip()
#
#
# def has_numbers(input_string):
#     return bool(re.search(r"\d", input_string))
#
#
# def del_multichar(input_string, chars):
#     for ch in chars:
#         input_string = input_string.replace(ch, "")
#     return input_string
#
#
# def merge_two_dicts(x, y):
#     z = x.copy()  # start with x's keys and values
#     z.update(y)  # modifies z with y's keys and values & returns None
#     return z
#
#
def early_stopping(
    log_value, best_value, stopping_step, expected_order="acc", flag_step=100
):
    # early stopping strategy:
    assert expected_order in ["acc", "dec"]

    if (expected_order == "acc" and log_value >= best_value) or (
        expected_order == "dec" and log_value <= best_value
    ):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print(
            "Early stopping is trigger at step: {} log:{}".format(flag_step, log_value)
        )
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


# def freeze(model):
#     for param in model.parameters():
#         param.requires_grad = False
#     return model
#
#
# def unfreeze(model):
#     for param in model.parameters():
#         param.requires_grad = True
#     return model


def print_dict(dic):
    """print dictionary using specified format

    example: {"a": 1, "b": 2}
    output:
            "a": 1
            "b": 2
    """
    print("\n".join("{:10s}: {}".format(key, values) for key, values in dic.items()))

#
# ##########################################################AL
#
# import numpy as np
# import scipy.sparse as sp
# import warnings
# from sklearn.metrics import f1_score
# import torch
# import math
#
#
# def preprocess_features(features):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     rowsum = np.array(features.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = np.diag(r_inv)
#     features = r_mat_inv.dot(features)
#     return features
#
# def column_normalize(tens):
#     ret = tens - tens.mean(axis=0)
#     return ret
#
# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
#
# def preprocess_adj(adj):
#     """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
#     adj_add_diag=adj + sp.eye(adj.shape[0])
#     adj_normalized = normalize_adj(adj_add_diag)
#     return adj_normalized.astype(np.float32) #sp.coo_matrix(adj_unnorm)
#
# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)
#
# ##=========================================================================
#
# def accuracy(y_pred, labels):
#     if len(labels.size())==1:
#         y_pred = y_pred.max(1)[1].type_as(labels)
#         y_pred=y_pred.cpu().detach().numpy()
#         labels=labels.cpu().numpy()
#
#
#     elif len(labels.size())==2:
#         # print("rawy_pred",y_pred)
#         y_pred=(y_pred > 0.).cpu().detach().numpy()
#         labels=labels.cpu().numpy()
#
#     # y_pred = np.zeros_like(y_pred)
#
#     # print("y_pred",y_pred[:10,:])
#     # print("labels",labels[:10,:])
#     # exit()
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         mic,mac=f1_score(labels, y_pred, average="micro"), f1_score(labels, y_pred, average="macro")
#     return mic,mac
#
#
#
# ##==========================================================================
#
#
# ##==========================================================================
#
#
# class AverageMeter(object):
#     def __init__(self,name='',ave_step=10):
#         self.name = name
#         self.ave_step = ave_step
#         self.history =[]
#         self.history_extrem = None
#         self.S=5
#
#     def update(self,data):
#         if data is not None:
#             self.history.append(data)
#
#     def __call__(self):
#         if len(self.history) == 0:
#             value =  None
#         else:
#             cal=self.history[-self.ave_step:]
#             value = sum(cal)/float(len(cal))
#         return value
#
#     def should_save(self):
#         if len(self.history)>self.S*2 and sum(self.history[-self.S:])/float(self.S)> sum(self.history[-self.S*2:])/float(self.S*2):
#             if self.history_extrem is None :
#                 self.history_extrem =sum(self.history[-self.S:])/float(self.S)
#                 return False
#             else:
#                 if self.history_extrem < sum(self.history[-self.S:])/float(self.S):
#                     self.history_extrem = sum(self.history[-self.S:])/float(self.S)
#                     return True
#                 else:
#                     return False
#         else:
#             return False
#
#
# #===========================================================
#
# def inspect_grad(model):
#     name_grad = [(x[0], x[1].grad) for x in model.named_parameters() if x[1].grad is not None]
#     name, grad = zip(*name_grad)
#     assert not len(grad) == 0, "no layer requires grad"
#     mean_grad = [torch.mean(x) for x in grad]
#     max_grad = [torch.max(x) for x in grad]
#     min_grad = [torch.min(x) for x in grad]
#     logger.info("name {}, mean_max min {}".format(name,list(zip(mean_grad, max_grad, min_grad))))
#
# def inspect_weight(model):
#     name_weight = [x[1] for x in model.named_parameters() if x[1].grad is not None]
#     print("network_weight:{}".format(name_weight))
#
#
# #==============================================================
#
# def common_rate(counts,prediction,seq):
#     summation = counts.sum(dim=1, keepdim=True)
#     squaresum = (counts ** 2).sum(dim=1, keepdim=True)
#     ret = (summation ** 2 - squaresum) / (summation * (summation - 1)+1)
#     # print("here1")
#     equal_rate=counts[seq,prediction].reshape(-1,1)/(summation+1)
#     # print(ret,equal_rate)
#     return ret,equal_rate
#
#
# def logprob2Prob(logprobs):
#     probs = F.softmax(logprobs, dim=2)
#     return probs
#
# def normalizeEntropy(entro,classnum): #this is needed because different number of classes will have different entropy
#     maxentro = np.log(float(classnum))
#     entro = entro/maxentro
#     return entro
#
# def prob2Logprob(probs):
#     logprobs = torch.log(probs,np.float)
#     return logprobs
#
# def perc(input):
#     # the biger valueis the biger result is
#     numnode = input.size(-2)
#     res = torch.argsort(torch.argsort(input, dim=-2), dim=-2) / float(numnode)
#     return res
