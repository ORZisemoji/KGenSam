import torch

import numpy as np
from tqdm import tqdm
import math
import torch.nn as nn
from torch.autograd import Variable
from utils import cuda_

def mean_std(L):
    if type(L)!=type([]):
        L=L.tolist()
    m=sum(L)/float(len(L))
    bias=[(x-m)**2 for x in L]
    std=math.sqrt(sum(bias)/float(len(L)-1))
    return [float(m),float(std)]


def degprocess(deg):
    # deg = torch.log(1+deg)
    #return deg/20.
    return torch.clamp_max(deg / 20., 1.)

def localdiversity(probs,adj,deg):
    # probs.size: torch.Size([10, 2708, 7])
    indices = adj.coalesce().indices()
    N =adj.size()[0]
    classnum = probs.size()[-1]
    maxentro = np.log(float(classnum))
    edgeprobs = probs[:,indices.transpose(0,1),:]
    headprobs = edgeprobs[:,:,0,:]
    tailprobs = edgeprobs[:,:,1,:]
    kl_ht = (torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*tailprobs,dim=-1) - \
             torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*tailprobs,dim=-1)).transpose(0,1)
    kl_th = (torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*headprobs,dim=-1) - \
             torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*headprobs,dim=-1)).transpose(0,1)
    sparse_output_kl_ht = torch.sparse.FloatTensor(indices,kl_ht,size=torch.Size([N,N,kl_ht.size(-1)]))
    sparse_output_kl_th = torch.sparse.FloatTensor(indices,kl_th,size=torch.Size([N,N,kl_th.size(-1)]))
    sum_kl_ht = torch.sparse.sum(sparse_output_kl_ht,dim=1).to_dense().transpose(0,1)
    sum_kl_th = torch.sparse.sum(sparse_output_kl_th,dim=1).to_dense().transpose(0,1)
    mean_kl_ht = sum_kl_ht/(deg+1e-10)
    mean_kl_th = sum_kl_th/(deg+1e-10)
    # normalize
    mean_kl_ht = mean_kl_ht / mean_kl_ht.max(dim=1, keepdim=True).values
    mean_kl_th = mean_kl_th / mean_kl_th.max(dim=1, keepdim=True).values
    return mean_kl_ht,mean_kl_th

def entropy(tens,multilabel=False):
    if multilabel:#Todo
        reverse=1-tens
        ent_1= -torch.log(torch.clamp(tens, min=1e-7)) * tens
        ent_2= -torch.log(torch.clamp(reverse,min=1e-7))*reverse
        ent=ent_1+ent_2
        entropy=torch.mean(ent,dim=1)
    else:
        assert type(tens)==torch.Tensor and len(tens.size())==3,"calculating entropy of wrong size"
        entropy = - torch.log(torch.clamp(tens, min=1e-7)) * tens
        entropy = torch.sum(entropy, dim=2)
    return entropy

