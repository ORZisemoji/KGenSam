import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math

import sys

sys.path.append('/home/mengyuan/KGenSam')
from utils import cuda_

class FM(nn.Module):
    def __init__(self, args_config,cfg):
        super(FM, self).__init__()

        self.args_config = args_config
        self.cfg=cfg

        self.n_users = self.cfg.n_users
        self.n_items = self.cfg.n_items
        self.PAD_IDX1=self.cfg.PAD_IDX1
        self.n_attributes = self.cfg.n_attributes
        self.PAD_IDX2=self.cfg.PAD_IDX2

        self.emb_size = self.args_config.emb_size  # dimensions
        self.reg = self.args_config.reg
        self.ip = self.args_config.ip
        self.dr = self.args_config.dr
        self.dropout2 = nn.Dropout(p=self.dr)  # dropout ratio

        # _______ User embedding + Item embedding
        self.ui_emb = nn.Embedding(self.PAD_IDX1+1, self.emb_size, sparse=False)

        # _______ Attribute embedding
        # self.attri_emb = nn.Embedding(self.PAD_IDX2+1, self.emb_size, sparse=False)
        self.attri_emb = nn.Embedding(self.PAD_IDX2+1, self.emb_size, padding_idx=self.PAD_IDX2,sparse=False)

        # _______ Scala Bias _______
        self.Bias = nn.Parameter(torch.randn(1).normal_(0, 0.01), requires_grad=True)

        self.init_weight()

    def init_weight(self):
        self.ui_emb.weight.data.normal_(0, 0.01)
        self.attri_emb.weight.data.normal_(0, self.ip)
        # # _______set the padding to zero _____
        self.attri_emb.weight.data[self.n_attributes,:]=0

    def forward(self, ui_pair,preference_index):
        '''
        param: ui_pair；a list of user ID and busi ID
        '''
        if ui_pair is not None and preference_index is not None:
            feature_matrix_ui = self.ui_emb(ui_pair)  # (bs, 19, emb_size), 19 is the largest padding
            feature_matrix_preference = self.attri_emb(preference_index)  # (bs, 2, emb_size)
            feature_matrix = torch.cat((feature_matrix_ui, feature_matrix_preference), dim=1)
        elif ui_pair is not None and preference_index is None:
            feature_matrix_ui = self.ui_emb(ui_pair)  # (bs, 19, emb_size), 19 is the largest padding
            feature_matrix = feature_matrix_ui
        elif ui_pair is None and preference_index is not None:
            feature_matrix_preference = self.attri_emb(preference_index)  # (bs, 2, emb_size)
            feature_matrix = feature_matrix_preference
        else:
            return 0


        # _______ make a clone _______
        feature_matrix_clone = feature_matrix.clone()

        # _________ sum_square part _____________
        summed_features_embedding_squared = feature_matrix.sum(dim=1, keepdim=True) ** 2  # (bs, 1, emb_size)

        # _________ square_sum part _____________
        squared_sum_features_embedding = (feature_matrix * feature_matrix).sum(dim=1, keepdim=True)  # (bs, 1, emb_size)

        # ________ FM __________
        fm = 0.5 * (summed_features_embedding_squared - squared_sum_features_embedding)  # (bs, 1, emb_size)

        # Optional: remove the inter-group interaction
        # ***---***
        if ui_pair is not None and preference_index is not None:
            new_non_zero_2 = feature_matrix_preference
            summed_features_embedding_squared_new_2 = new_non_zero_2.sum(dim=1, keepdim=True) ** 2
            squared_sum_features_embedding_new_2 = (new_non_zero_2 * new_non_zero_2).sum(dim=1, keepdim=True)
            newFM_2 = 0.5 * (summed_features_embedding_squared_new_2 - squared_sum_features_embedding_new_2)
            fm = (fm - newFM_2)
        # ***---***

        fm = self.dropout2(fm)  # (bs, 1, emb_size)

        Bilinear = fm.sum(dim=2, keepdim=False)  # (bs, 1)
        result = Bilinear + self.Bias  # (bs, 1)

        return result, feature_matrix_clone

    
    ########### for kg negative-sampler
    def translate_sampler_data_for_fm(self, users, items):
        # print('len(users):{};len(items):{}'.format(len(users),len(items)))
        # print('users:{}'.format(users))
        assert len(users)==len(items)
        num=len(users)
        def trans_to_intlist(tens):
            tmp=list(tens)
            for i in range(len(tmp)):
                tmp[i]=int(tmp[i])
            return tmp
        u_list=trans_to_intlist(users)
        i_list=trans_to_intlist(items)
        ui_list = []
        for i in range(num):
            ui_list.append(torch.LongTensor([u_list[i],i_list[i]]))
        ui_list = pad_sequence(ui_list, batch_first=True, padding_value=self.PAD_IDX1)
        # print("ui_list.shape:{}".format(ui_list.shape))
        # print("ui_list:{}".format(ui_list))
        return cuda_(ui_list)

    def translate_sampler_data_for_fmrank(self, user, items):
        users=[user]*len(items)
        users=torch.tensor(users)
        return self.translate_sampler_data_for_fm(users, items)

    def get_loss(self, user, pos_item, neg_item):
        # bpr loss
        lsigmoid = nn.LogSigmoid()
        pos_list=self.translate_sampler_data_for_fm(user,pos_item)
        neg_list=self.translate_sampler_data_for_fm(user,neg_item)

        result_pos, nonzero_matrix_pos = self.forward(pos_list,None)  # (bs, 1)
        result_neg, nonzero_matrix_neg = self.forward(neg_list,None)
        diff = (result_pos - result_neg)
        bpr_loss = - lsigmoid(diff).sum(dim=0)

        # regularization
        nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)
        nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)
        reg_loss = (self.reg * nonzero_matrix_pos_).sum(dim=0)+(self.reg * nonzero_matrix_neg_).sum(dim=0)

        return bpr_loss, reg_loss

    def get_reward(self, user, pos_item, neg_item,k_step):
        # print('get_reward')
        # print('user.shape:{}'.format(user.shape))
        # print('pos_item.shape:{}'.format(pos_item.shape))
        # print('neg_item.shape:{}'.format(neg_item.shape))

        u_n_pair1=self.translate_sampler_data_for_fm(user,neg_item[0,...].squeeze())
        neg_scores1,_=self.forward(u_n_pair1,None)
        n_p_pair1=self.translate_sampler_data_for_fm(neg_item[0,...].squeeze(),pos_item)
        ij1,_=self.forward(n_p_pair1,None)
        reward = neg_scores1 + ij1

        if  k_step==2:
            u_n_pair2=self.translate_sampler_data_for_fm(user,neg_item[1,...].squeeze())
            neg_scores2,_=self.forward(u_n_pair2,None)

            n_p_pair2 = self.translate_sampler_data_for_fm(neg_item[1, ...].squeeze(), pos_item)
            ij2, _ = self.forward(n_p_pair2,None)

            reward = reward + neg_scores2+ ij2

        return reward

    def rank(self, users, items):
        # u_e = self.ui_emb[users]
        # i_e = self.ui_emb[items]
        #
        # u_e = u_e.unsqueeze(dim=1)
        # ranking = torch.sum(u_e * i_e, dim=2)
        users,items=list(users.cpu()),list(items.cpu())
        # print('list(users):{}'.format(users))
        alluser_ranking=[]
        for i in range(len(users)): # bs=1024
            u_i_pair=self.translate_sampler_data_for_fmrank(users[i],items[i])
            u_ranking,_=self.forward(u_i_pair,None)
            alluser_ranking.append(list(u_ranking.squeeze()))
        # print('alluser_ranking:{}'.format(alluser_ranking))
        ranking=torch.tensor(alluser_ranking)
        ranking = ranking.squeeze()

        return ranking

    def __str__(self):
        return "recommender using BPRFM, embedding size {}".format(
            self.emb_size
        )
