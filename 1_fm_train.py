
import sys
sys.path.append('/home/mengyuan/KGenSam')
from utils import cuda_,set_random_seed
set_random_seed()


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
import time
from torch.nn.utils.rnn import pad_sequence


sys.path.append('/home/mengyuan/KGenSam/FM')
from factorization_machine import FM
from fm_item_evaluate import evaluate_item
from fm_attribute_evaluate import  evaluate_attribute

sys.path.append('/home/mengyuan/KGenSam/configuration')
from base_config import bcfg

sys.path.append('/home/mengyuan/KGenSam/data_hepler')
from data_in import load_fm_sample,load_fm_model
from data_out import creat_fm_model_logfile,save_fm_model,save_final_fm_model,save_fm_model_embedding,save_fm_model_log,save_final_fm_model_embedding





def translate_pickle_to_data(pickle_file, iter_, bs, pickle_file_length,updatefeatureemb):
    '''
    user_pickle = pickle_file[0]
    item_p_pickle = pickle_file[1]
    i_neg1_pickle = pickle_file[2]
    i_neg2_pickle = pickle_file[3]
    preference_pickle = pickle_file[4]
    '''
    left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)

    pos_list, neg_list, new_neg_list,preference_list,new_preference_list = [], [], [],[],[]

    I = pickle_file[0][left:right]
    II = pickle_file[1][left:right]
    III = pickle_file[2][left:right]
    IV = pickle_file[3][left:right]
    V = pickle_file[4][left:right]

    residual_feature, neg_feature = None, None
    if updatefeatureemb == 1:
        feature_range = np.arange(bcfg.n_attributes).tolist()
        residual_feature, neg_feature = [], []
        for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
            gt_feature = bcfg.item_dict[str(item_p_pickle)]['feature_index']
            this_residual_feature = list(set(bcfg.item_dict[str(item_p_pickle)]['feature_index']) - set(preference_pickle))
            remain_feature = list(set(feature_range) - set(gt_feature))
            this_neg_feature = np.random.choice(remain_feature, len(this_residual_feature))
            residual_feature.append(torch.LongTensor(this_residual_feature))
            neg_feature.append(torch.LongTensor(this_neg_feature))
        residual_feature = pad_sequence(residual_feature, batch_first=True, padding_value=bcfg.PAD_IDX2)
        neg_feature = pad_sequence(neg_feature, batch_first=True, padding_value=bcfg.PAD_IDX2)

    i = 0
    index_none = list()
    for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
        pos_list.append(torch.LongTensor([user_pickle, item_p_pickle + len(bcfg.user_list)]))
        neg_list.append(torch.LongTensor([user_pickle, i_neg1_pickle + len(bcfg.user_list)]))
        preference_list.append(torch.LongTensor(preference_pickle))

        if i_neg2_pickle is None:
            index_none.append(i)
        i += 1

    i = 0
    for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
        if i in index_none:
            i += 1
            continue
        new_neg_list.append(torch.LongTensor([user_pickle, i_neg2_pickle + len(bcfg.user_list)]))
        new_preference_list.append(torch.LongTensor(preference_pickle))
        i += 1

    pos_list = pad_sequence(pos_list, batch_first=True, padding_value=bcfg.PAD_IDX1)
    neg_list = pad_sequence(neg_list, batch_first=True, padding_value=bcfg.PAD_IDX1)
    new_neg_list = pad_sequence(new_neg_list, batch_first=True, padding_value=bcfg.PAD_IDX1)
    preference_list = pad_sequence(preference_list, batch_first=True, padding_value=bcfg.PAD_IDX2)
    new_preference_list = pad_sequence(new_preference_list, batch_first=True, padding_value=bcfg.PAD_IDX2)

    if updatefeatureemb != 0:
        return cuda_(pos_list), cuda_(neg_list), cuda_(new_neg_list), cuda_(preference_list), cuda_(new_preference_list), index_none, cuda_(residual_feature), cuda_(neg_feature)
    else:
        return cuda_(pos_list), cuda_(neg_list), cuda_(new_neg_list), cuda_(preference_list), cuda_(new_preference_list), index_none, residual_feature, neg_feature





def train(filename,model, bs, max_epoch, optimizer1, optimizer2, optimizer3, reg, observe,pretrain,updatefeatureemb,updateuseremb):
    model.train()
    lsigmoid = nn.LogSigmoid()
    reg_float = float(reg.data.cpu().numpy()[0])


    for epoch in range(max_epoch):
        # _______ Do the evaluation _______
        if epoch % observe == 0 and epoch > 0:
            print('Evaluating on item prediction')
            evaluate_item(model, epoch, filename)
            print('Evaluating on feature similarity')
            evaluate_attribute(model, epoch, filename)

        tt = time.time()
        pickle_file = load_fm_sample(mode='train', epoch=epoch % 50)

        print('Open pickle file takes {} seconds'.format(time.time() - tt))
        pickle_file_length = len(pickle_file[0])

        model.train()

        # user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle = zip(*pickle_file[left:right])
        mix = list(zip(pickle_file[0], pickle_file[1], pickle_file[2], pickle_file[3], pickle_file[4]))
        random.shuffle(mix)
        I, II, III, IV, V = zip(*mix)
        new_pk_file = [I, II, III, IV, V]

        start = time.time()
        print('\nStarting {} epoch'.format(epoch))
        epoch_loss = 0
        epoch_loss_2 = 0
        max_iter = int(pickle_file_length / float(bs))

        for iter_ in range(max_iter):
            if iter_ > 1 and iter_ % 100 == 0:
                print('--')
                print('Takes {} seconds to finish {}% of this epoch'.format(str(time.time() - start),
                                                                            float(iter_) * 100 / max_iter))
                print('loss is: {}'.format(float(epoch_loss) / (bs * iter_)))
                print('iter_:{} Bias grad norm: {}, Static grad norm: {}, Preference grad norm: {}'.format(iter_,torch.norm(model.Bias.grad),torch.norm(model.ui_emb.weight.grad),torch.norm(model.attri_emb.weight.grad)))

            pos_list,neg_list,new_neg_list, preference_list, new_preference_list, index_none, residual_feature, neg_feature \
                = translate_pickle_to_data(new_pk_file, iter_, bs, pickle_file_length,updatefeatureemb)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            result_pos, nonzero_matrix_pos = model(pos_list,preference_list)  # (bs, 1), (bs, 2, emb_size)

            result_neg, nonzero_matrix_neg = model(neg_list,preference_list)
            diff = (result_pos - result_neg)
            loss = - lsigmoid(diff).sum(dim=0)  # The Minus is crucial is


            # The second type of negative sample
            new_result_neg, new_nonzero_matrix_neg = model(new_neg_list,new_preference_list)
            # Reason for this is that, sometimes the sample is missing, so we have to also omit that in result_pos
            T = cuda_(torch.tensor([]))
            for i in range(bs):
                if i in index_none:
                    continue
                T = torch.cat([T, result_pos[i]], dim=0)
            T = T.view(T.shape[0], -1)
            assert T.shape[0] == new_result_neg.shape[0]
            diff = T - new_result_neg
            if loss is not None:
                loss += - lsigmoid(diff).sum(dim=0)
            else:
                loss = - lsigmoid(diff).sum(dim=0)


            # regularization
            if reg_float != 0:
                nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                loss += (reg * nonzero_matrix_pos_).sum(dim=0)
                loss += (reg * nonzero_matrix_neg_).sum(dim=0)
            epoch_loss += loss.data
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            if updatefeatureemb == 1:
                # updating feature embedding
                # we try to optimize
                A = model.attri_emb(preference_list)[..., :]
                user_emb = model.ui_emb(pos_list[:, 0])[..., :].unsqueeze(dim=1).detach()
                if updateuseremb == 1:
                    A = torch.cat([A, user_emb], dim=1)
                B = model.attri_emb(residual_feature)[..., :]
                C = model.attri_emb(neg_feature)[..., :]

                D = torch.matmul(A, B.transpose(2, 1))
                E = torch.matmul(A, C.transpose(2, 1))

                p_vs_residual = D.view(D.shape[0], -1, 1)
                p_vs_neg = E.view(E.shape[0], -1, 1)

                p_vs_residual = p_vs_residual.sum(dim=1)
                p_vs_neg = p_vs_neg.sum(dim=1)
                diff = (p_vs_residual - p_vs_neg)
                temp = - lsigmoid(diff).sum(dim=0)
                loss = temp
                epoch_loss_2 += temp.data

                if iter_ % 1000 == 0 and iter_ > 0:
                    print('2ND iter_:{} preference grad norm: {}'.format(iter_, torch.norm(model.attri_emb.weight.grad)))
                    print('2ND loss is: {}'.format(float(epoch_loss_2) / (bs * iter_)))

                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()


            # These line is to make an alert on console when we meet gradient explosion.
            if iter_ > 0 and iter_ % 1 == 0:
                if torch.norm(model.ui_emb.weight.grad) > 100 :
                    print('iter_:{} Bias grad norm: {}, F-bias grad norm: {}'.format(iter_,torch.norm(model.Bias.grad),torch.norm(model.ui_emb.weight.grad)))

            # Uncomment this to use clip gradient norm (but currently we don't need)
            # clip_grad_norm_(model.ui_emb.weight, 5000)
            # clip_grad_norm_(model.attri_emb.weight, 5000)

        print('epoch loss: {}'.format(epoch_loss / pickle_file_length))
        print('epoch loss 2: {}'.format(epoch_loss_2 / pickle_file_length))

        if epoch % 50 == 0 and epoch > 0:
            print('FM Epoch：{} ; start saving FM model.'.format(epoch))
            save_fm_model(model=model, filename=filename, epoch=epoch)
            print('FM Epoch：{} ; start saving model embedding.'.format(epoch))
            save_fm_model_embedding(model=model, filename=filename,epoch=epoch)

        train_len = len(pickle_file[0])
        save_fm_model_log(filename=filename, epoch=epoch, epoch_loss=epoch_loss
                          , epoch_loss_2=epoch_loss_2, train_len=train_len)

    print('FM FINAL {} Epoch; start saving FM model.'.format(max_epoch))
    save_final_fm_model(model=model, filename='FM-model')
    print('FM FINAL {} Epoch; start saving model embedding.'.format(max_epoch))
    save_final_fm_model_embedding(model=model, filename='FM-model')



if __name__ == '__main__':

    """initialize args"""
    A=bcfg.get_FM_parser()

    A.lr=0.01
    A.flr=0.001
    A.optim='SGD'
    A.reg=0.002

    # """fix the random seed"""
    # A.seed = 2021
    # random.seed(A.seed)
    # np.random.seed(A.seed)
    # torch.manual_seed(A.seed)
    # torch.cuda.manual_seed(A.seed)

    if A.pretrain_epoch == 0:
        # means no pretrain
        model = FM(args_config=A,cfg=bcfg)
    else:
        model = FM(args_config=A,cfg=bcfg)
        file_name = 'V4-FM-lr-{}-flr-{}-optim-{}-pretrain_epoch-{}-max_epoch-{}-updatefeatureemb-{}-updateuseremb-{}-seed-{}'.format(
            A.lr, A.flr, A.optim, A.pretrain_epoch, A.max_epoch, A.updatefeatureemb, A.updateuseremb, A.seed)
        model_dict = load_fm_model(file_name, epoch=A.pretrain_epoch)
        model.load_state_dict(model_dict)
    cuda_(model)

    param1, param2 = list(), list()
    param3 = list()

    i = 0
    for name, param in model.named_parameters():
        print(name, param)
        if i == 0:
            param1.append(param)  # bias层
        else:
            param2.append(param)  # ui和attri的embdedding层
        if i == 2:
            param3.append(param)  # attri的embdedding层
        i += 1

    print('param1 is: {}, shape:{}\nparam2 is: {}, shape: {}\nparam3 is: {}, shape: {}\n'.format(param1,
                                                                                                 [param.shape for param
                                                                                                  in param1], param2,
                                                                                                 [param.shape for param
                                                                                                  in param2], param3,
                                                                                                 [param.shape for param
                                                                                                  in param3]))

    if A.optim == 'SGD':
        optimizer1 = torch.optim.SGD(param1, lr=A.lr, weight_decay=0.1)
        optimizer2 = torch.optim.SGD(param2, lr=A.lr)
        optimizer3 = torch.optim.SGD(param3, lr=A.flr)
    if A.optim == 'Ada':
        optimizer1 = torch.optim.Adagrad(param1, lr=A.lr, weight_decay=A.decay)
        optimizer2 = torch.optim.Adagrad(param2, lr=A.lr, weight_decay=A.decay)
        optimizer3 = torch.optim.Adagrad(param3, lr=A.flr, weight_decay=A.decay)

    reg_ = torch.Tensor([A.reg])
    reg_ = torch.autograd.Variable(reg_, requires_grad=False)
    reg_ = cuda_(reg_)

    file_name = 'V4-FM-lr-{}-flr-{}-optim-{}-pretrain_epoch-{}-max_epoch-{}-updatefeatureemb-{}-updateuseremb-{}-seed-{}'.format(A.lr, A.flr, A.optim,A.pretrain_epoch,A.max_epoch,A.updatefeatureemb,A.updateuseremb, A.seed)



    # creat new log file
    creat_fm_model_logfile(file_name)

    train(file_name,model, A.bs, A.max_epoch, optimizer1, optimizer2,optimizer3, reg_,A.observe, A.pretrain_epoch,A.updatefeatureemb,A.updateuseremb)


