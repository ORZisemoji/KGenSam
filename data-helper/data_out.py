import pickle
import numpy as np
import random
import torch
import os
import sys
sys.path.append('/home/mengyuan/AUM-V4/configuration')
from base_config import bcfg
sys.path.append('/home/mengyuan/AUM-V4/data_hepler')
from data_in import load_fm_model,load_pretrain_fm_model

'''fm'''
def save_fm_model(model, filename, epoch):
    model_file = '{}/FM-model/epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    if not os.path.isdir(bcfg.log_root+ '/FM-model/'):
        os.makedirs(bcfg.log_root+ '/FM-model/')
    torch.save(model.state_dict(), model_file)
    print('FM Model in epoch {} saved at {}!'.format(epoch,model_file))

def save_final_fm_model(model, filename='FM-model'):
    model_file = '{}/FM-model/{}.pt'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+ '/FM-model/'):
        os.makedirs(bcfg.log_root+ '/FM-model/')
    torch.save(model.state_dict(), model_file)
    print('FINAL FM Model saved at {}!'.format(model_file))

def save_fm_model_embedding(model, filename,epoch):
    model_dict = load_fm_model(filename, epoch)
    model.load_state_dict(model_dict)
    print('Model loaded successfully!')
    ui_emb = model.ui_emb.weight[..., :-1].data.cpu().numpy()
    attri_emb = model.attri_emb.weight[..., :-1].data.cpu().numpy()
    print('ui_size:{}'.format(ui_emb.shape[0]))
    print('fea_size:{}'.format(attri_emb.shape[0]))
    embeds = {
        'ui_emb': ui_emb,
        'feature_emb': attri_emb
    }
    path = '{}/FM-model/embeds-epoch-{}-{}.pkl'.format(bcfg.log_root,epoch,filename)
    if not os.path.isdir(bcfg.log_root+ '/FM-model/'):
        os.makedirs(bcfg.log_root+ '/FM-model/')
    with open(path, 'wb') as f:
        pickle.dump(embeds, f)
        print('FM Embedding in epoch {} saved at {}!'.format(epoch,path))

def save_final_fm_model_embedding(model, filename):
    model_dict = load_pretrain_fm_model(filename)
    model.load_state_dict(model_dict)
    print('FINAL Model loaded successfully!')
    ui_emb = model.ui_emb.weight[..., :-1].data.cpu().numpy()
    attri_emb = model.attri_emb.weight[..., :-1].data.cpu().numpy()
    print('ui_size:{}'.format(ui_emb.shape[0]))
    print('fea_size:{}'.format(attri_emb.shape[0]))
    embeds = {
        'ui_emb': ui_emb,
        'feature_emb': attri_emb
    }
    path = '{}/FM-model/embeds-{}.pkl'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+ '/FM-model/'):
        os.makedirs(bcfg.log_root+ '/FM-model/')
    with open(path, 'wb') as f:
        pickle.dump(embeds, f)
        print('FINAL FM Embedding saved at {}!'.format(path))


# log
def creat_fm_model_logfile(filename):
    PATH = '{}/FM-log/{}.txt'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+'/FM-log/'):
        os.makedirs(bcfg.log_root+'/FM-log/')
    with open(PATH, 'w') as f:
        f.write('\n')
    print('New create fm model log file :{} !!!'.format(PATH))

def save_fm_model_log(filename, epoch, epoch_loss, epoch_loss_2, train_len):
    PATH = '{}/FM-log/{}.txt'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+ '/FM-log/'):
        os.makedirs(bcfg.log_root + '/FM-log/')
    with open(PATH, 'a') as f:
        f.write('Starting {} epoch\n'.format(epoch))
        f.write('training loss 1: {}\n'.format(epoch_loss / train_len))
        f.write('training loss 2: {}\n'.format(epoch_loss_2 / train_len))
        f.write('=============================\n')

def save_fm_model_log_evaluate(filename, epoch, auc_mean, auc_median, evaluate_subject):
    # evaluate_subject='item' or 'attribute'
    PATH ='{}/FM-log/{}.txt'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+ '/FM-log/'):
        os.makedirs(bcfg.log_root + '/FM-log/')
    with open(PATH, 'a') as f:
        f.write('validating {} epoch on {} prediction\n'.format(epoch,evaluate_subject))
        f.write('auc mean: {}\n'.format(auc_mean))
        f.write('auc median: {}\n'.format(auc_median))


'''negative sampler'''

def save_negative_sampler_model(model, epoch, filename='Negative_Sampler_model'):
    if bcfg.data_feature_two_layer:
        model_file = '{}/Negative-Sampler-model/2_layer_feature-epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    else:
        model_file = '{}/Negative-Sampler-model/epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    if not os.path.isdir(bcfg.log_root+ '/Negative-Sampler-model/'):
        os.makedirs(bcfg.log_root+ '/Negative-Sampler-model/')
    torch.save(model.state_dict(), model_file)
    print('Negative Sampler Model in epoch {} saved at {}!'.format(epoch,model_file))

def save_final_negative_sampler_model(model, filename='Negative_Sampler_model'):
    if bcfg.data_feature_two_layer:
        model_file = '{}/Negative-Sampler-model/2_layer_feature-{}.pt'.format(bcfg.log_root,filename)
    else:
        model_file = '{}/Negative-Sampler-model/{}.pt'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+ '/Negative-Sampler-model/'):
        os.makedirs(bcfg.log_root+ '/Negative-Sampler-model/')
    torch.save(model.state_dict(), model_file)
    print('FINAL Negative Sampler Model saved at {}!'.format(model_file))


'''active sampler'''
def save_active_sampler_model(model, epoch, filename='Active_Sampler_model'):
    if bcfg.data_feature_two_layer:
        model_file = '{}/Active-Sampler-model/2_layer_feature-epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    else:
        model_file = '{}/Active-Sampler-model/epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    if not os.path.isdir(bcfg.log_root+ '/Active-Sampler-model/'):
        os.makedirs(bcfg.log_root+ '/Active-Sampler-model/')
    torch.save(model.state_dict(), model_file)
    print('Active Sampler Model in epoch {} saved at {}!'.format(epoch,model_file))

def save_final_active_sampler_model(model, filename='Active_Sampler_model'):
    if bcfg.data_feature_two_layer:
        model_file = '{}/Active-Sampler-model/2_layer_feature-{}.pt'.format(bcfg.log_root,filename)
    else:
        model_file = '{}/Active-Sampler-model/{}.pt'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+ '/Active-Sampler-model/'):
        os.makedirs(bcfg.log_root+ '/Active-Sampler-model/')
    torch.save(model.state_dict(), model_file)
    print('FINAL Active Sampler Model saved at {}!'.format(model_file))

'''conversational rl'''
def save_rl_model(model, epoch, filename='Conversational_Policy_model'):
    if not bcfg.alSampler*bcfg.negSampler:
        filename='alSampler-{}-negSampler-{}-'.format(bcfg.alSampler,bcfg.negSampler)+filename
    if bcfg.data_feature_two_layer:
        model_file = '{}/Conversational-Policy-model/2_layer_feature-epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    else:
        model_file = '{}/Conversational-Policy-model/epoch-{}-{}.pt'.format(bcfg.log_root,epoch,filename)
    if not os.path.isdir(bcfg.log_root+'/Conversational-Policy-model/'):
        os.makedirs(bcfg.log_root+'/Conversational-Policy-model/')
    torch.save(model.state_dict(), model_file)
    print('Conversational Policy model in epoch {} saved at {}'.format(epoch,model_file))

def save_final_rl_model(model, filename='Conversational_Policy_model'):
    if not bcfg.alSampler * bcfg.negSampler:
        filename = 'alSampler-{}-negSampler-{}-'.format(bcfg.alSampler, bcfg.negSampler) + filename
    if bcfg.data_feature_two_layer:
        model_file = '{}/Conversational-Policy-model/2_layer_feature-{}.pt'.format(bcfg.log_root,filename)
    else:
        model_file = '{}/Conversational-Policy-model/{}.pt'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+'/Conversational-Policy-model/'):
        os.makedirs(bcfg.log_root+'/Conversational-Policy-model/')
    torch.save(model.state_dict(), model_file)
    print('FINAL Conversational Policy model saved at {}'.format(model_file))

# log
def creat_rl_model_logfile(filename):
    if bcfg.data_feature_two_layer:
        filename = '2_layer_feature-'+filename
    PATH = '{}/Conversational-Policy-log/{}.txt'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+'/Conversational-Policy-log/'):
        os.makedirs(bcfg.log_root+'/Conversational-Policy-log/')
    with open(PATH, 'w') as f:
        f.write('\n')
    print('New create RL model log file :{} !!!'.format(PATH))

def save_rl_mtric(filename, epoch, SR, spend_time, mode='train'):
    if bcfg.data_feature_two_layer:
        filename = '2_layer_feature-'+filename
    PATH = '{}/Conversational-Policy-log/{}.txt'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+'/Conversational-Policy-log/'):
        os.makedirs(bcfg.log_root+'/Conversational-Policy-log/')
    if mode == 'train':
        with open(PATH, 'a') as f:
            f.write('===========Train===============\n')
            f.write('Starting {} user epochs\n'.format(epoch))
            f.write('training SR@5: {}\n'.format(SR[0]))
            f.write('training SR@10: {}\n'.format(SR[1]))
            f.write('training SR@15: {}\n'.format(SR[2]))
            f.write('training Avg@T: {}\n'.format(SR[3]))
            f.write('Spending time: {}\n'.format(spend_time))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))
    elif mode == 'valid':
        with open(PATH, 'a') as f:
            f.write('===========Valid===============\n')
            f.write('Validating {} user tuples\n'.format(epoch))
            f.write('Validating SR@5: {}\n'.format(SR[0]))
            f.write('Validating SR@10: {}\n'.format(SR[1]))
            f.write('Validating SR@15: {}\n'.format(SR[2]))
            f.write('Validating Avg@T: {}\n'.format(SR[3]))
            f.write('Validating time: {}\n'.format(spend_time))
            f.write('================================\n')
    elif mode == 'test':
        with open(PATH, 'a') as f:
            f.write('===========Test===============\n')
            f.write('Testing {} user tuples\n'.format(epoch))
            f.write('Testing SR@5: {}\n'.format(SR[0]))
            f.write('Testing SR@10: {}\n'.format(SR[1]))
            f.write('Testing SR@15: {}\n'.format(SR[2]))
            f.write('Testing Avg@T: {}\n'.format(SR[3]))
            f.write('Testing time: {}\n'.format(spend_time))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))

def save_rl_model_log(filename, epoch, epoch_loss, train_len):
    if bcfg.data_feature_two_layer:
        filename = '2_layer_feature-'+filename
    PATH = '{}/Conversational-Policy-log/{}.txt'.format(bcfg.log_root,filename)
    if not os.path.isdir(bcfg.log_root+'/Conversational-Policy-log/'):
        os.makedirs(bcfg.log_root+'/Conversational-Policy-log/')
    with open(PATH, 'a') as f:
        f.write('Starting {} epoch\n'.format(epoch))
        f.write('training loss : {}\n'.format(epoch_loss / train_len))
        # f.write('1000 loss: {}\n'.format(loss_1000))


