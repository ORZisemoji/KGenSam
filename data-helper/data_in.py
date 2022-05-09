import pickle
import json
import numpy as np
import random
import torch
import os
import sys
sys.path.append('/home/mengyuan/AUM-V4/configuration')
from base_config import bcfg
sys.path.append('/home/mengyuan/AUM-V4/data-helper')
from data_loader import FM_Data_loader

'''fm'''
def load_fm_sample(mode, epoch=0):
    # dir='FM-sample-data'
    # if mode == 'train':
    #     sample_file = '{}/data/{}/sample_fm_data_{}-{}.pkl'.format(bcfg.data_root,dir,mode, epoch)
    # if mode == 'valid':
    #     sample_file = '{}/data/{}/sample_fm_data_{}.pkl'.format(bcfg.data_root,dir,mode)
    # with open(sample_file, 'rb') as f:
    #     sample_data = pickle.load(f)
    # return sample_data
    return FM_Data_loader(mode, epoch)

def load_fm_model(filename, epoch):
    model_file = '{}/FM-model/epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    model_dict = torch.load(model_file)
    print('{}.pt Model load at {}'.format(filename,model_file))
    return model_dict

def load_pretrain_fm_model(filename='FM-model'):
    model_file = '{}/FM-model/{}.pt'.format(bcfg.log_root,filename)
    model_dict = torch.load(model_file)
    # print('{}.pt Model load at {}'.format(filename,model_file))
    return model_dict

#
# def load_embed(dataset, epoch):
#     path = TMP_DIR[dataset] + '/FM-model-embeds/' + 'embeds-epoch-{}.pkl'.format(epoch)
#     with open(path, 'rb') as f:
#         embeds = pickle.load(f)
#         print('FM Epoch：{} Embedding load successfully!'.format(epoch))
#         return embeds
#

'''negative sampler'''

def load_ns_model(epoch,filename='Negative_Sampler_model'):
    #epoch-33-Negative_Sampler_model.pt
    if bcfg.data_feature_two_layer:
        model_file = '{}/Negative-Sampler-model/2_layer_feature-epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    else:
        model_file = '{}/Negative-Sampler-model/epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    model_dict = torch.load(model_file)
    print('{}.pt Model load at {}'.format(filename,model_file))
    return model_dict

def load_pretrain_ns_model(filename='Negative_Sampler_model'):
    if bcfg.data_feature_two_layer:
        model_file = '{}/Negative-Sampler-model/2_layer_feature-{}.pt'.format(bcfg.log_root,filename)
    else:
        model_file = '{}/Negative-Sampler-model/{}.pt'.format(bcfg.log_root,filename)
    model_dict = torch.load(model_file)
    print('{}.pt Model load at {}'.format(filename,model_file))
    return model_dict

'''active sampler'''

def load_as_model(epoch,filename='Active_Sampler_model'):
    #epoch-33-active_Sampler_model.pt
    if bcfg.data_feature_two_layer:
        model_file = '{}/Active-Sampler-model/2_layer_feature-epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    else:
        model_file = '{}/Active-Sampler-model/epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    model_dict = torch.load(model_file)
    print('{}.pt Model load at {}'.format(filename,model_file))
    return model_dict

def load_pretrain_as_model(filename='Active_Sampler_model'):
    if bcfg.data_feature_two_layer:
        model_file = '{}/Active-Sampler-model/2_layer_feature-{}.pt'.format(bcfg.log_root,filename)
    else:
        model_file = '{}/Active-Sampler-model/{}.pt'.format(bcfg.log_root,filename)
    model_dict = torch.load(model_file)
    print('{}.pt Model load at {}'.format(filename,model_file))
    return model_dict

'''conversational rl'''

def load_rl_model(epoch, filename='Conversational_Policy_model'):
    if bcfg.data_feature_two_layer:
        model_file = '{}/Conversational-Policy-model/2_layer_feature-epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    else:
        model_file = '{}/Conversational-Policy-model/epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    model_dict = torch.load(model_file)
    print('{}.pt Model load at {}'.format(filename,model_file))
    return model_dict

def load_rl_data(mode):
    print('---------------------- load_rl_data -----------------------------------')
    if mode == 'train':
        mydict = bcfg._train_user_to_items
        print('train_data: load conversational-policy train data')
    elif mode == 'valid':
        mydict = bcfg._valid_user_to_items
        print('valid_data: load conversational-policy valid data')
    elif mode == 'test':
        mydict = bcfg._test_user_to_items
        print('test_data: load conversational-policy test data')
    return mydict
