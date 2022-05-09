# BB-8 and R2-D2 are best friends.

import json
import numpy as np
import os
import random
from utils import cuda_
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tkinter import _flatten
from collections import Counter

import sys
sys.path.append('/home/mengyuan/KGenSam/active-sampler')
from al_policy import ALagent
sys.path.append('/home/mengyuan/KGenSam/data-helper')
from data_in import load_rl_data
sys.path.append('/home/mengyuan/KGenSam/configuration')
from base_config import bcfg

from collections import Counter
import numpy as np
from random import randint
import json
import random
import tqdm
import time
import copy

class user():
    def __init__(self, user_id, busi_id):
        self.user_id = user_id
        self.busi_id = busi_id
        self.pos_items_id=bcfg._user_to_items[str(user_id)]

    def response(self,rec_list):
        data = dict()
        data['accept']=False
        data['ranking'] = -1
        data['total'] = len(rec_list)
        data['accepted_item_list'] = []
        data['rejected_item_list'] = []
        for rec_item in rec_list:
            if rec_item in self.pos_items_id:
                if not data['accept']:
                    data['ranking'] = rec_list.index(rec_item) + 1
                data['accept']=True
                data['accepted_item_list'].append(rec_item)
        data['rejected_item_list'] = []
        for i in rec_list:
            if i not in data['accepted_item_list']:
                data['rejected_item_list'].append(i)
        return data
    # end def


class BinaryRecommendEnv(object):
    def __init__(self, kg, args,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler,mode):
        self.data_name = bcfg.data_name
        self.A=args
        self.state_command = args.state_command
        self.mode = mode
        self.seed = args.seed
        self.max_turn = args.max_turn    #MAX_TURN
        self.cand_len_size = args.cand_len_size
        self.reward_option=args.reward_option

        self.kg = kg

        self.attr_state_num = bcfg.n_attributes
        self.feature_length = bcfg.n_attributes
        self.user_length = bcfg.n_users
        self.item_length = bcfg.n_items

        self.reset_FM_model(pretrain_FM_model)
        self.neg_Sampler=copy.deepcopy(pretrain_neg_sampler)
        self.al_Sampler=copy.deepcopy(pretrain_al_sampler)

        # action parameters
        self.rec_num = 10
        self.ent_way = args.entropy_method #  active_policy , entropy  or weight_entropy

        # user's profile
        self.reachable_feature = []   # user reachable feature
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = []   # candidate items
        self.user_rej_items = [] # user rejected item which recommended by agent

        #user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        #  the number of conversation in current step
        self.cur_node_set = []     # maybe a node or a node set  /   normally save feature node
        # state veactor
        self.user_embed = None
        self.conver_his = []    #conversation_history
        self.cand_len = []    #the number of candidate items  [binary ]
        self.attr_ent = []  # attribute entropy

        self.ui_dict = load_rl_data(mode= self.mode)  # u list[i]
        self.user_weight_dict = dict()

        #init user_dict
        if self.mode == 'train':
            self.__user_weight_dict_init__() # init self.user_weight_dict
        elif self.mode in ['test','valid']:
            self.ui_array = None    # u-i array [ [userID1, itemID1], ...,[userID2, itemID2]]
            self.__test_tuple_generate__()
            self.test_num = 0


        self.action_space = 2


        self.state_space_dict = {
            1: self.max_turn + self.cand_len_size + self.attr_state_num + self.ui_embeds.shape[1],
            2: self.attr_state_num,  # attr_ent
            3: self.max_turn,  #conver_his
            4: self.cand_len_size,  #cand_item
            5: self.ui_embeds.shape[1], # user_embedding
            6: self.cand_len_size + self.attr_state_num + self.max_turn, #attr_ent + conver_his + cand_item
            7: self.cand_len_size + self.max_turn,
        }
        self.state_space = self.state_space_dict[self.state_command]
        if self.reward_option==0: # copy scpr
            self.reward_dict = {
                'ask_suc': 0.01,
                'ask_fail': -0.1,
                'rec_suc': 1,
                'rec_fail': -0.1,
                'until_T': -0.3,      # MAX_Turn
                'cand_none': -0.1
            }
        elif self.reward_option==1:# ask more
            self.reward_dict = {
                # 'ask_suc': 0.01,
                'ask_suc': 0.1,
                'ask_fail': -0.1,
                'rec_suc': 1,
                # 'rec_fail': -0.1,
                'rec_fail': -1,
                'until_T': -0.3,      # MAX_Turn
                'cand_none': -0.1
            }
        elif self.reward_option==11:# ask more comparably
            self.reward_dict = {
                'ask_suc': 0.5,
                'ask_fail': -0.1,
                'rec_suc': 1,
                'rec_fail': -0.1,
                'until_T': -0.3,      # MAX_Turn
                'cand_none': -0.1
            }
        elif self.reward_option==2: # rec more
            self.reward_dict = {
                'ask_suc': 0.01,
                'ask_fail': -0.1,
                'rec_suc': 1,
                'rec_fail': -0.01,
                'until_T': -0.3,      # MAX_Turn
                'cand_none': -0.1
            }
        elif self.reward_option==3: # copy ear
            self.reward_dict = {
                'ask_suc': 0.11,
                'ask_fail': 0.01,
                'rec_suc': 1.01,
                'rec_fail': 0.01,
                'until_T': -0.3,      # MAX_Turn
                'cand_none': -0.1
            }


        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_scu': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict()   # This dict is used to calculate entropy
        print('reward_dict:{}'.format(self.reward_dict))

    def __get_fm_embeds__(self):
        ui_emb = self.cur_FM_model.ui_emb.weight[..., :-1].data.cpu().numpy()
        attri_emb = self.cur_FM_model.attri_emb.weight[..., :-1].data.cpu().numpy()
        return ui_emb,attri_emb


    def __user_weight_dict_init__(self):   #Calculate the weight of the number of interactions per user
        ui_nums = 0
        for items in self.ui_dict.values():
            ui_nums += len(items)
        for user_str in self.ui_dict.keys():
            user_id = int(user_str)
            self.user_weight_dict[user_id] = len(self.ui_dict[user_str])/ui_nums
        print('user_dict init successfully!')

    def __test_tuple_generate__(self):
        ui_list = []
        for user_str, items in self.ui_dict.items():
            user_id = int(user_str)
            for item_id in items:
                ui_list.append([user_id, item_id])
        self.ui_array = np.array(ui_list)
        np.random.shuffle(self.ui_array)

    def _get_state(self):
        if self.state_command == 1:
            state = [self.user_embed, self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
        elif self.state_command == 2: #attr_ent
            state = self.attr_ent
            state = list(_flatten(state))
        elif self.state_command == 3: #conver_his
            state = self.conver_his
            state = list(_flatten(state))
        elif self.state_command == 4: #cand_len
            state = self.cand_len
            state = list(_flatten(state))
        elif self.state_command == 5:  #user_embedding
            state = self.user_embed
            state = list(_flatten(state))
        elif self.state_command == 6: #attr_ent + conver_his + cand_len
            state = [self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
        elif self.state_command == 7: #conver_his + cand_len
            state = [self.conver_his, self.cand_len]
            state = list(_flatten(state))
        return state


    def _build_sampler_graph(self):
        n_nodes=self.kg.entity_range[1] + 1
        edge_threshold=self.neg_Sampler.params.edge_threshold
        graph=copy.deepcopy(self.kg.ckg_graph)

        adj_matrix = torch.zeros(n_nodes, edge_threshold * 2)
        edge_matrix = torch.zeros(n_nodes, edge_threshold)

        """sample neighbors for each node"""
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            # print(len(neighbors))
            # break
            if len(neighbors) >= edge_threshold:
                sampled_edge = random.sample(neighbors, edge_threshold)
                edges = copy.deepcopy(sampled_edge)
            else:
                neg_id = random.sample(
                    range(self.kg.item_range[0], self.kg.item_range[1] + 1),
                    edge_threshold - len(neighbors),
                    )
                node_id = [node] * (edge_threshold - len(neighbors))
                sampled_edge = neighbors + neg_id
                edges = neighbors + node_id

            """concatenate sampled edge with random edge"""
            sampled_edge += random.sample(
                range(self.kg.item_range[0], self.kg.item_range[1] + 1), edge_threshold
            )

            adj_matrix[node] = torch.tensor(sampled_edge, dtype=torch.long)
            edge_matrix[node] = torch.tensor(edges, dtype=torch.long)

        if torch.cuda.is_available():
            adj_matrix = adj_matrix.cuda().long()
            edge_matrix = edge_matrix.cuda().long()

        return adj_matrix, edge_matrix

    def rebuild_sampler_graph(self):
        self.adj_matrix, self.edge_matrix = self._build_sampler_graph()


    def reset_user(self,user_id,target_item):
        self.rebuild_sampler_graph()
        #init  user_id  item_id  cur_step   cur_node_set
        self.cur_conver_step = 0   #reset cur_conversation step
        self.cur_node_set = []

        # init user's profile
        print('-----------reset state vector------------')
        self.user_id = user_id
        self.target_item = target_item
        self.user=user(user_id,target_item)
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        self.reachable_feature = []  # user reachable feature in cur_step
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.user_rej_items = []
        self.cand_items = list(range(self.item_length))

        # init  state vector
        self.user_embed = self.ui_embeds[self.user_id].tolist()  # init user_embed   np.array---list
        self.conver_his = [0] * self.max_turn  # conversation_history
        self.cand_len = [self.feature_length >> d & 1 for d in range(self.cand_len_size)][::-1]  #Binary representation of candidate set length
        self.attr_ent = [0] * self.attr_state_num  # attribute entropy

        #init user prefer feature
        self._updata_reachable_feature(start='user')  # self.reachable_feature = []
        self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1

        print('=== init user prefer feature: {}'.format(self.cur_node_set))
        self._update_cand_items(acc_feature=self.cur_node_set, rej_feature=[])
        self._update_feature_entropy()  #update entropy
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))

        # Sort reachable features according to the entropy of features
        reach_fea_score = self._feature_score()
        max_ind_list = []

        max_score = max(reach_fea_score)
        max_ind = reach_fea_score.index(max_score)
        reach_fea_score[max_ind] = 0
        max_ind_list.append(max_ind)

        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        return self._get_state()

    def reset_FM_model(self,cur_fm_model):
        self.cur_FM_model=copy.deepcopy(cur_fm_model)
        """load fm epoch"""
        self.ui_embeds,self.feature_emb = self.__get_fm_embeds__()
        """set fm optim"""
        param_all, param_attri = list(), list()

        i = 0
        for name, param in self.cur_FM_model.named_parameters():
            param_all.append(param)
            if i == 2:
                param_attri.append(param)  # attri的embdedding层
            i += 1
        self.recommender_optimizer = torch.optim.Adam(param_all, lr=self.A.rlr,weight_decay=self.cur_FM_model.args_config.decay)
        self.recommender_optimizer_attri = torch.optim.Adam(param_attri, lr=self.A.rlr, weight_decay=self.cur_FM_model.args_config.decay)



    def update_FM_model(self):
        """update FM_model"""
        """This function is used to update the pretrained FM model """

        pos_items = list(set(bcfg._user_to_items[str(self.user_id)]) - set([self.target_item]))
        neg_items = self.user_rej_items[-10:]
        self.recommender_optimizer.zero_grad()
        """1 update FM_model using negtive item provided by negative-sampler"""
        if self.A.negSampler:
            batch_data_s=dict()
            batch_data_s["pos_i_id"]=cuda_(torch.tensor([i+bcfg.n_users for i in pos_items]))
            batch_data_s["u_id"] = cuda_(torch.tensor([self.user_id]*len(pos_items)))
            random_neg = list(set(bcfg.item_list) - set(bcfg._train_user_to_items[str(self.user_id)])
                              - set(bcfg._valid_user_to_items[str(self.user_id)])
                              - set(bcfg._test_user_to_items[str(self.user_id)]))
            neg = random.sample(random_neg, len(pos_items))
            batch_data_s["neg_i_id"]=cuda_(torch.tensor([i+bcfg.n_users for i in neg]))

            selected_neg_items_list, _ = self.neg_Sampler(batch_data_s, self.adj_matrix, self.edge_matrix)
            selected_neg_items = selected_neg_items_list[-1, :]

            def process_neg_items(users,neg,selected_neg_items):
                def build_train_data(train_mat):
                    num_user = max(train_mat.keys()) + 1
                    num_true = max([len(i) for i in train_mat.values()])
                    train_data = torch.zeros(num_user, num_true)
                    for i in train_mat.keys():
                        true_list = train_mat[i]
                        true_list += [-1] * (num_true - len(true_list))
                        train_data[i] = torch.tensor(true_list, dtype=torch.long)
                    return cuda_(train_data)

                train_mat = self.kg.train_user_dict
                train_data = build_train_data(train_mat)
                train_set = train_data[users]
                in_train = torch.sum(
                    selected_neg_items.unsqueeze(1) == train_set.long(), dim=1
                ).byte()
                selected_neg_items[in_train] = neg[in_train]
                return selected_neg_items

            selected_neg_items=process_neg_items(batch_data_s["u_id"],batch_data_s["neg_i_id"],selected_neg_items)
            base_loss_batch, reg_loss_batch = self.cur_FM_model.get_loss(batch_data_s["u_id"], batch_data_s["pos_i_id"], selected_neg_items)
            loss_batch = base_loss_batch + reg_loss_batch

            loss_batch.backward()
            self.recommender_optimizer.step()

        """2 update FM_model using negtive item provided by user-stimulator"""
        self.cur_FM_model.train()
        bs = 32
        # to remove all ground truth interacted items ...
        random_neg = list(set(bcfg.item_dict) - set(bcfg._train_user_to_items[str(self.user_id)])
                          - set(bcfg._valid_user_to_items[str(self.user_id)])
                          - set(bcfg._test_user_to_items[str(self.user_id)]))

        pos_items = pos_items + random.sample(random_neg,
                                              len(pos_items))  # add some random negative samples to avoid overfitting
        neg_items = neg_items + random.sample(random_neg,
                                              len(neg_items))  # add some random negative samples to avoid overfitting

        # _______ Form Pair _______
        if not neg_items:
            # print('update fm using negtive item provided by user-stimulator failed - empty neg list')
            return
        # pos_neg_pairs = list()
        # for p_item in pos_items:
        #     for n_item in neg_items:
        #         pos_neg_pairs.append((p_item, n_item))

        pos_neg_pairs = list()

        num = int(bs / len(pos_items)) + 1
        pos_items = pos_items * num

        for p_item in pos_items:
            n_item = random.choice(neg_items)
            pos_neg_pairs.append((p_item, n_item))
        random.shuffle(pos_neg_pairs)

        max_iter = int(len(pos_neg_pairs) / bs)

        reg_ = torch.Tensor([self.cur_FM_model.reg])
        reg_ = torch.autograd.Variable(reg_, requires_grad=False)
        reg_ = cuda_(reg_)
        reg = reg_

        lsigmoid = nn.LogSigmoid()

        def get_batch_data(self, pos_neg_pairs, bs, iter_, acc_p, rej_p):
            # this function is used for
            # Get batched data for updating FM model

            left = iter_ * bs
            right = min((iter_ + 1) * bs, len(pos_neg_pairs))
            pos_list, pos_list2, neg_list, neg_list2 = list(), list(), list(), list()
            for instance in pos_neg_pairs[left: right]:
                # instance[0]: pos item, instance[1] neg item
                pos_list.append(torch.LongTensor([self.user_id, instance[0] + bcfg.n_users]))
                neg_list.append(torch.LongTensor([self.user_id, instance[1] + bcfg.n_users]))
            # end for
            pos_preference_list = torch.LongTensor(acc_p).expand(len(pos_list), len(acc_p))
            neg_preference_list = torch.LongTensor(rej_p).expand(len(neg_list), len(rej_p))

            pos_list = pad_sequence(pos_list, batch_first=True, padding_value=bcfg.PAD_IDX1)
            pos_list2 = pos_preference_list

            neg_list = pad_sequence(neg_list, batch_first=True, padding_value=bcfg.PAD_IDX1)
            neg_list2 = neg_preference_list

            return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2)
            # end def

        for iter_ in range(max_iter):
            pos_list, pos_list2, neg_list, neg_list2 = get_batch_data(pos_neg_pairs, bs, iter_, self.user_acc_feature,self.user_rej_feature)
            result_pos, nonzero_matrix_pos = self.cur_FM_model(pos_list,pos_list2)
            result_neg, nonzero_matrix_neg = self.cur_FM_model(neg_list,neg_list2)
            diff = (result_pos - result_neg)
            loss = - lsigmoid(diff).sum(dim=0)

            nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)
            nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)
            loss += (reg * nonzero_matrix_pos_).sum(dim=0)
            loss += (reg * nonzero_matrix_neg_).sum(dim=0)

            self.recommender_optimizer.zero_grad()
            loss.backward()
            self.recommender_optimizer.step()
        # end for


    def step(self, action):   #action:0  ask   action:1  recommend   setp=MAX_TURN  done
        done = 0
        print('---------------Turn:{}-------------'.format(self.cur_conver_step))

        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')
            done = 1
        elif action == 0:   #ask feature
            print('-->action: ask features')
            reward, done, acc_feature, rej_feature = self._ask_update()  #update user's profile:  user_acc_feature & user_rej_feature
            self._update_cand_items(acc_feature, rej_feature)   #update cand_items

            if len(acc_feature):   # can reach new feature：  update current node and reachable_feature
                self.cur_node_set = acc_feature
                self._updata_reachable_feature(start='feature')  # update user's profile: reachable_feature

            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_rej_feature))
            print('user_acc_feature:{}'.format(self.user_acc_feature))
            print('user_rej_feature:{}'.format(self.user_rej_feature))

            if self.state_command in [1, 2, 6, 7]:  # update attr_ent
                self._update_feature_entropy()
            if len(self.reachable_feature) != 0:  # if reachable_feature == 0 :cand_item= 1

                if self.ent_way in ['entropy','weight_entropy']:
                    reach_fea_score = self._feature_score()  # compute feature score

                    max_score = max(reach_fea_score)
                    max_ind = reach_fea_score.index(max_score)
                    max_fea_id = self.reachable_feature[max_ind]
                else:#  active_policy

                    al_agent = ALagent(self.user_id,self.target_item,self.user_acc_feature,self.al_Sampler, None, self.kg.ckg_graph,test=True)

                    max_fea_id = al_agent.select_fuzzy_sample(self.cur_FM_model,self.reachable_feature,self.user_rej_feature+self.user_acc_feature,self.user_acc_feature)

                    max_ind = self.reachable_feature.index(max_fea_id)
                # print('max_fea_id:{}'.format(max_fea_id))
                # print('self.reachable_feature:{}'.format(self.reachable_feature))
                self.reachable_feature.pop(max_ind)
                self.reachable_feature.insert(0, max_fea_id)

        elif action == 1:  #recommend items
            #select topk candidate items to recommend
            cand_item_score = self._item_score()
            item_score_tuple = list(zip(self.cand_items, cand_item_score))
            sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
            self.cand_items, cand_item_score = zip(*sort_tuple)


            #===================== rec update=========
            reward, done = self._recommend_updata()
            #========================================
            if reward == 1:
                print('-->Recommend successfully!')
            else:
                for i in range(self.A.update_fm_count):
                    self.update_FM_model()
                print('----- update current FM {} times -----'.format(self.A.update_fm_count))
                if self.state_command in [1, 2, 6, 7]:  # update attr_ent
                    self._update_feature_entropy()
                print('-->Recommend fail !')

        self.cur_conver_step += 1
        return self._get_state(), reward, done

    def _updata_reachable_feature(self, start='feature'):
        #self.reachable_feature = []
        if start == 'user':
            user_like_random_fea = random.choice(bcfg.item_dict[str(self.target_item)]['feature_index'])
            self.user_acc_feature.append(user_like_random_fea) #update user acc_fea
            self.cur_node_set = [user_like_random_fea]

            next_reachable_feature = []
            for cur_node in self.cur_node_set:
                fea_belong_items = bcfg.feature_dict[str(cur_node)]['item_index_list']  # A-I
                # fea_like_users = list(self.kg.G['feature'][cur_node]['like'])  # A-U

                # if self.data_name == 'lastfm':
                #     # update reachable feature
                #     user_friends = self.kg.G['user'][self.user_id]['friends']
                #     cand_fea_like_users = list(set(fea_like_users) & set(user_friends))
                #     for user_id in cand_fea_like_users:  # A-U-A  # U in [friends]
                #         next_reachable_feature.append(list(self.kg.G['user'][user_id]['like']))
                #     next_reachable_feature = list(set(_flatten(next_reachable_feature)))

                cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))

                for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items]
                    next_reachable_feature.append(bcfg.item_dict[str(item_id)]['feature_index'])
                next_reachable_feature = list(set(_flatten(next_reachable_feature)))
            self.reachable_feature = next_reachable_feature  # init reachable_feature

        elif start == 'feature':
            next_reachable_feature = []
            for cur_node in self.cur_node_set:
                fea_belong_items = bcfg.feature_dict[str(cur_node)]['item_index_list'] # A-I
                # fea_like_users = list(self.kg.G['feature'][cur_node]['like'])   #A-U
                #
                # if self.data_name == 'lastfm':
                #     # update reachable feature
                #     user_friends = self.kg.G['user'][self.user_id]['friends']
                #     cand_fea_like_users = list(set(fea_like_users) & set(user_friends))
                #     for user_id in cand_fea_like_users:  # A-U-A  # U in [friends]
                #         next_reachable_feature.append(list(self.kg.G['user'][user_id]['like']))
                #     next_reachable_feature = list(set(_flatten(next_reachable_feature)))

                cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))
                for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items]
                    next_reachable_feature.append(bcfg.item_dict[str(item_id)]['feature_index'])
                next_reachable_feature = list(set(_flatten(next_reachable_feature)))
            self.reachable_feature = next_reachable_feature


    def _feature_score(self):
        reach_fea_score = []
        for feature_id in self.reachable_feature:
            score = self.attr_ent[feature_id]
            reach_fea_score.append(score)
        return reach_fea_score

    def _item_score(self):
        # cand_item_score = []
        # for item_id in self.cand_items:
        #     item_embed = self.ui_embeds[self.user_length + item_id]
        #     score = 0
        #     score += np.inner(np.array(self.user_embed), item_embed)
        #     prefer_embed = self.feature_emb[self.user_acc_feature, :]  #np.array (x*64)
        #     for i in range(len(self.user_acc_feature)):
        #         score += np.inner(prefer_embed[i], item_embed)
        #     cand_item_score.append(score)
        # return cand_item_score
        self.cur_FM_model.eval()
        mini_ui_pair = np.zeros((len(self.cand_items), 2))
        for index, itemID in enumerate(self.cand_items):
            mini_ui_pair[index, :] = [self.user_id, itemID + bcfg.n_users]
        mini_ui_pair = torch.from_numpy(mini_ui_pair).long()
        mini_ui_pair = cuda_(mini_ui_pair)

        static_preference_index = torch.LongTensor(self.user_acc_feature).expand(len(self.cand_items), len(self.user_acc_feature))  # candidate_list, given preference
        static_preference_index = cuda_(static_preference_index)

        static_score, _ = self.cur_FM_model(mini_ui_pair, static_preference_index)
        static_score = static_score.detach().cpu().numpy()

        cand_item_score=static_score.reshape(-1).tolist()
        return cand_item_score


    def _ask_update(self):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0
        # TODO datafram!     groundTruth == target_item features
        # feature_groundtrue = self.kg.G['item'][self.target_item]['belong_to']
        feature_groundtrue = bcfg.item_dict[str(self.target_item)]['feature_index']

        remove_acced_reachable_fea = self.reachable_feature.copy()  # copy reachable_feature

        acc_feature = list(set(remove_acced_reachable_fea[:1]) & set(feature_groundtrue))
        rej_feature = list(set(remove_acced_reachable_fea[:1]) - set(acc_feature))

        #update user_acc_feature & user_rej_feature
        self.user_acc_feature.append(acc_feature)
        self.user_acc_feature = list(set(_flatten(self.user_acc_feature)))
        self.user_rej_feature.append(rej_feature)
        self.user_rej_feature = list(set(_flatten(self.user_rej_feature)))


        if len(acc_feature):
            reward = self.reward_dict['ask_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']   #update conver_his
        else:
            reward = self.reward_dict['ask_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  #update conver_his

        if self.cand_items == []:  #candidate items is empty
            done = 1
            reward = self.reward_dict['cand_none']
        return reward, done, acc_feature, rej_feature

    def _update_cand_items(self, acc_feature, rej_feature):
        if len(acc_feature):    #accept feature
            print('=== ask acc: update cand_items')
            for feature_id in acc_feature:
                # feature_items = self.kg.G['feature'][feature_id]['belong_to']
                feature_items = bcfg.feature_dict[str(feature_id)]['item_index_list']
                self.cand_items = set(self.cand_items) & set(feature_items)   #  itersection
            self.cand_items = list(self.cand_items)

        self.cand_len = [len(self.cand_items) >>d & 1 for d in range(self.cand_len_size)][::-1]  # binary

    def _recommend_updata(self):
        print('-->action: recommend items')
        recom_items = self.cand_items[: self.rec_num]    # TOP k item to recommend
        print('recom_items:{}'.format(recom_items))
        user_response=self.user.response(rec_list=recom_items)
        print('user_response：\n {}'.format(user_response))
        # if user_response['accept']:
        if self.target_item in recom_items:
            reward = self.reward_dict['rec_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu'] #update state vector: conver_his
            done = 1
        else:
            reward = self.reward_dict['rec_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']  #update state vector: conver_his
            if len(self.cand_items) > self.rec_num:
                self.cand_items = self.cand_items[self.rec_num:]  #update candidate items
            self.cand_len = [len(self.cand_items) >> d & 1 for d in range(self.cand_len_size)][::-1]  #  binary
            done = 0
        return reward, done

    def _update_feature_entropy(self):
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            for item_id in self.cand_items:
                # cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))
                cand_items_fea_list.append(list(bcfg.item_dict[str(item_id)]['feature_index']))
            cand_items_fea_list = list(_flatten(cand_items_fea_list))
            self.attr_count_dict = dict(Counter(cand_items_fea_list))
            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                p2 = 1.0 - p1
                if p1 == 1:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent
        elif self.ent_way == 'weight_entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            cand_item_score = self._item_score()
            cand_item_score_sig = self.sigmoid(cand_item_score)  # sigmoid(score)
            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(bcfg.item_dict[str(item_id)]['feature_index'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            sum_score_sig = sum(cand_item_score_sig)

            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig
                p2 = 1.0 - p1
                if p1 == 1 or p1 <= 0:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent

    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()



class EnumeratedRecommendEnv(object):
    def __init__(self, kg, args,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler,mode):
        self.data_name = bcfg.data_name
        self.A=args
        self.state_command = args.state_command
        self.mode = mode
        self.seed = args.seed
        self.max_turn = args.max_turn    #MAX_TURN
        self.cand_len_size = args.cand_len_size

        self.kg = kg

        self.attr_state_num = bcfg.n_big_attributes
        self.feature_length = bcfg.n_attributes
        self.user_length = bcfg.n_users
        self.item_length = bcfg.n_items

        self.reset_FM_model(pretrain_FM_model)
        self.neg_Sampler=copy.deepcopy(pretrain_neg_sampler)
        self.al_Sampler=copy.deepcopy(pretrain_al_sampler)

        # action parameters
        self.rec_num = 10
        self.ent_way = args.entropy_method #  active_policy , entropy  or weight_entropy

        # user's profile
        self.reachable_feature = []   # user reachable large_feature
        self.user_acc_feature = []  # user accepted large_feature which asked by agent
        self.user_rej_feature = []  # user rejected large_feature which asked by agent
        self.acc_small_fea = []
        self.rej_small_fea = []
        self.cand_items = []   # candidate items
        self.user_rej_items = [] # user rejected item which recommended by agent


        #user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        #  the number of conversation in current step
        self.cur_node_set = []     #maybe a node or a node set  /   normally save large_feature node
        # state veactor
        self.user_embed = None
        self.conver_his = []    #conversation_history
        self.cand_len = []    #the number of candidate items  [ binary]
        self.attr_ent = []  # attribute entropy

        self.ui_dict = load_rl_data(mode= self.mode)  # u list[i]
        self.user_weight_dict = dict()

        #init user_dict
        if self.mode == 'train':
            self.__user_weight_dict_init__() # init self.user_weight_dict
        elif self.mode in ['test','valid']:
            self.ui_array = None    # u-i array [ [userID1, itemID1], ...,[userID2, itemID2]]
            self.__test_tuple_generate__()
            self.test_num = 0

        self.action_space = 2

        self.state_space_dict = {
            1: self.max_turn + self.cand_len_size + self.attr_state_num + self.ui_embeds.shape[1],
            2: self.attr_state_num,  # attr_ent
            3: self.max_turn,  #conver_his
            4: self.cand_len_size,  #cand_item
            5: self.ui_embeds.shape[1], # user_embedding
            6: self.cand_len_size + self.attr_state_num + self.max_turn, #attr_ent + conver_his + cand_item
            7: self.cand_len_size + self.max_turn,
            8: self.cand_len_size + self.attr_state_num + self.max_turn,

        }
        self.state_space = self.state_space_dict[self.state_command]
        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.1,
            'until_T': -0.3,      # MAX_Turn
            'cand_none': -0.1
        }
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_scu': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict()   # This dict is used to calculate entropy
        print('reward_dict:{}'.format(self.reward_dict))

    def __get_fm_embeds__(self):
        ui_emb = self.cur_FM_model.ui_emb.weight[..., :-1].data.cpu().numpy()
        attri_emb = self.cur_FM_model.attri_emb.weight[..., :-1].data.cpu().numpy()
        return ui_emb,attri_emb

    def __user_weight_dict_init__(self):   #Calculate the weight of the number of interactions per user
        ui_nums = 0
        for items in self.ui_dict.values():
            ui_nums += len(items)
        for user_str in self.ui_dict.keys():
            user_id = int(user_str)
            self.user_weight_dict[user_id] = len(self.ui_dict[user_str])/ui_nums
        print('user_dict init successfully!')

    def __test_tuple_generate__(self):
        ui_list = []
        for user_str, items in self.ui_dict.items():
            user_id = int(user_str)
            for item_id in items:
                ui_list.append([user_id, item_id])
        self.ui_array = np.array(ui_list)
        np.random.shuffle(self.ui_array)


    def _get_state(self):
        if self.state_command == 1:
            state = [self.user_embed, self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
        elif self.state_command == 2: #attr_ent
            state = self.attr_ent
            state = list(_flatten(state))
        elif self.state_command == 3: #conver_his
            state = self.conver_his
            state = list(_flatten(state))
        elif self.state_command == 4: #cand_len
            state = self.cand_len
            state = list(_flatten(state))
        elif self.state_command == 5:  #user_embedding
            state = self.user_embed
            state = list(_flatten(state))
        elif self.state_command == 6: #attr_ent + conver_his + cand_len
            state = [self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
        elif self.state_command == 7: #conver_his + cand_len
            state = [self.conver_his, self.cand_len]
            state = list(_flatten(state))
        # elif self.state_command == 8:
        #     state = [self.conver_his, self.attr_ent, self.cand_len]
        #     state = list(_flatten(state))
        return state


    def _build_sampler_graph(self):
        n_nodes=self.kg.entity_range[1] + 1
        edge_threshold=self.neg_Sampler.params.edge_threshold
        graph=copy.deepcopy(self.kg.ckg_graph)

        adj_matrix = torch.zeros(n_nodes, edge_threshold * 2)
        edge_matrix = torch.zeros(n_nodes, edge_threshold)

        """sample neighbors for each node"""
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            # print(len(neighbors))
            # break
            if len(neighbors) >= edge_threshold:
                sampled_edge = random.sample(neighbors, edge_threshold)
                edges = copy.deepcopy(sampled_edge)
            else:
                neg_id = random.sample(
                    range(self.kg.item_range[0], self.kg.item_range[1] + 1),
                    edge_threshold - len(neighbors),
                    )
                node_id = [node] * (edge_threshold - len(neighbors))
                sampled_edge = neighbors + neg_id
                edges = neighbors + node_id

            """concatenate sampled edge with random edge"""
            sampled_edge += random.sample(
                range(self.kg.item_range[0], self.kg.item_range[1] + 1), edge_threshold
            )

            adj_matrix[node] = torch.tensor(sampled_edge, dtype=torch.long)
            edge_matrix[node] = torch.tensor(edges, dtype=torch.long)

        if torch.cuda.is_available():
            adj_matrix = adj_matrix.cuda().long()
            edge_matrix = edge_matrix.cuda().long()

        return adj_matrix, edge_matrix

    def rebuild_sampler_graph(self):
        self.adj_matrix, self.edge_matrix = self._build_sampler_graph()

    def reset_user(self,user_id,target_item):
        self.rebuild_sampler_graph()
        #init  user_id  item_id  cur_step   cur_node_set
        self.cur_conver_step = 0   #reset cur_conversation step
        self.cur_node_set = []

        # init user's profile
        print('-----------reset state vector------------')
        self.user_id = user_id
        self.target_item = target_item
        self.user=user(user_id,target_item)
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        self.reachable_feature = []   # user reachable large_feature
        self.user_acc_feature = []  # user accepted large_feature which asked by agent
        self.user_rej_feature = []  # user rejected large_feature which asked by agent
        self.acc_small_fea = []
        self.rej_small_fea = []
        self.cand_items = list(range(self.item_length))

        # init  state vector
        self.user_embed = self.ui_embeds[self.user_id].tolist()  # init user_embed   np.array---list
        self.conver_his = [0] * self.max_turn  # conversation_history
        self.cand_len = [self.feature_length >>d & 1 for d in range(self.cand_len_size)][::-1]  #  Binary representation of candidate set length
        self.attr_ent = [0] * self.attr_state_num  #  attribute entropy

        # init user prefer feature
        self._updata_reachable_feature(start='user')  # self.reachable_feature = []
        self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1

        print('=== init user prefer large_feature: {}'.format(self.cur_node_set))
        self._update_cand_items(acc_feature=self.cur_node_set, rej_feature=[])
        self._update_feature_entropy()  # update entropy
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))



        #Sort reachable features according to the entropy of features
        reach_fea_score = self._feature_score()
        max_ind_list = []

        max_score = max(reach_fea_score)
        max_ind = reach_fea_score.index(max_score)
        reach_fea_score[max_ind] = 0
        max_ind_list.append(max_ind)

        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        return self._get_state()

    def reset_FM_model(self,cur_fm_model):
        self.cur_FM_model=copy.deepcopy(cur_fm_model)
        """load fm epoch"""
        self.ui_embeds,self.feature_emb = self.__get_fm_embeds__()
        """set fm optim"""
        param_all, param_attri = list(), list()

        i = 0
        for name, param in self.cur_FM_model.named_parameters():
            param_all.append(param)
            if i == 2:
                param_attri.append(param)  # attri的embdedding层
            i += 1
        self.recommender_optimizer = torch.optim.Adam(param_all, lr=self.A.rlr,weight_decay=self.cur_FM_model.args_config.decay)
        self.recommender_optimizer_attri = torch.optim.Adam(param_attri, lr=self.A.rlr, weight_decay=self.cur_FM_model.args_config.decay)

    def update_FM_model(self):
        """update FM_model"""
        """This function is used to update the pretrained FM model """

        pos_items = list(set(bcfg._user_to_items[str(self.user_id)]) - set([self.target_item]))
        neg_items = self.user_rej_items[-10:]
        self.recommender_optimizer.zero_grad()

        """1 update FM_model using negtive item provided by negative-sampler"""
        if self.A.negSampler:
            batch_data_s=dict()
            batch_data_s["pos_i_id"]=cuda_(torch.tensor([i+bcfg.n_users for i in pos_items]))
            batch_data_s["u_id"] = cuda_(torch.tensor([self.user_id]*len(pos_items)))
            random_neg = list(set(bcfg.item_list) - set(bcfg._train_user_to_items[str(self.user_id)])
                              - set(bcfg._valid_user_to_items[str(self.user_id)])
                              - set(bcfg._test_user_to_items[str(self.user_id)]))
            neg = random.sample(random_neg, len(pos_items))
            batch_data_s["neg_i_id"]=cuda_(torch.tensor([i+bcfg.n_users for i in neg]))

            selected_neg_items_list, _ = self.neg_Sampler(batch_data_s, self.adj_matrix, self.edge_matrix)
            selected_neg_items = selected_neg_items_list[-1, :]

            def process_neg_items(users,neg,selected_neg_items):
                def build_train_data(train_mat):
                    num_user = max(train_mat.keys()) + 1
                    num_true = max([len(i) for i in train_mat.values()])
                    train_data = torch.zeros(num_user, num_true)
                    for i in train_mat.keys():
                        true_list = train_mat[i]
                        true_list += [-1] * (num_true - len(true_list))
                        train_data[i] = torch.tensor(true_list, dtype=torch.long)
                    return cuda_(train_data)

                train_mat = self.kg.train_user_dict
                train_data = build_train_data(train_mat)
                train_set = train_data[users]
                in_train = torch.sum(
                    selected_neg_items.unsqueeze(1) == train_set.long(), dim=1
                ).byte()
                selected_neg_items[in_train] = neg[in_train]
                return selected_neg_items

            selected_neg_items=process_neg_items(batch_data_s["u_id"],batch_data_s["neg_i_id"],selected_neg_items)
            base_loss_batch, reg_loss_batch = self.cur_FM_model.get_loss(batch_data_s["u_id"], batch_data_s["pos_i_id"], selected_neg_items)
            loss_batch = base_loss_batch + reg_loss_batch

            loss_batch.backward()
            self.recommender_optimizer.step()

        """2 update FM_model using negtive item provided by user-stimulator"""
        self.cur_FM_model.train()
        bs = 32
        # to remove all ground truth interacted items ...
        random_neg = list(set(bcfg.item_dict) - set(bcfg._train_user_to_items[str(self.user_id)])
                          - set(bcfg._valid_user_to_items[str(self.user_id)])
                          - set(bcfg._test_user_to_items[str(self.user_id)]))

        pos_items = pos_items + random.sample(random_neg,
                                              len(pos_items))  # add some random negative samples to avoid overfitting
        neg_items = neg_items + random.sample(random_neg,
                                              len(neg_items))  # add some random negative samples to avoid overfitting

        # _______ Form Pair _______
        if not neg_items:
            # print('update fm using negtive item provided by user-stimulator failed - empty neg list')
            return
        # pos_neg_pairs = list()
        # for p_item in pos_items:
        #     for n_item in neg_items:
        #         pos_neg_pairs.append((p_item, n_item))

        pos_neg_pairs = list()

        num = int(bs / len(pos_items)) + 1
        pos_items = pos_items * num

        for p_item in pos_items:
            n_item = random.choice(neg_items)
            pos_neg_pairs.append((p_item, n_item))
        random.shuffle(pos_neg_pairs)

        max_iter = int(len(pos_neg_pairs) / bs)

        reg_ = torch.Tensor([self.cur_FM_model.reg])
        reg_ = torch.autograd.Variable(reg_, requires_grad=False)
        reg_ = cuda_(reg_)
        reg = reg_

        lsigmoid = nn.LogSigmoid()

        def get_batch_data(self, pos_neg_pairs, bs, iter_,acc_p,rej_p):
            # this function is used for
            # Get batched data for updating FM model

            left = iter_ * bs
            right = min((iter_ + 1) * bs, len(pos_neg_pairs))
            pos_list, pos_list2, neg_list, neg_list2 = list(), list(), list(), list()
            for instance in pos_neg_pairs[left: right]:
                #instance[0]: pos item, instance[1] neg item
                pos_list.append(torch.LongTensor([self.user_id, instance[0] + bcfg.n_users]))
                neg_list.append(torch.LongTensor([self.user_id, instance[1] + bcfg.n_users]))
            # end for
            pos_preference_list = torch.LongTensor(acc_p).expand(len(pos_list), len(acc_p))
            neg_preference_list = torch.LongTensor(rej_p).expand(len(neg_list), len(rej_p))

            pos_list = pad_sequence(pos_list, batch_first=True, padding_value=bcfg.PAD_IDX1)
            pos_list2 = pos_preference_list

            neg_list = pad_sequence(neg_list, batch_first=True, padding_value=bcfg.PAD_IDX1)
            neg_list2 = neg_preference_list

            return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2)
            # end def

        for iter_ in range(max_iter):
            pos_list, pos_list2, neg_list, neg_list2 = get_batch_data(pos_neg_pairs, bs, iter_,self.acc_small_fea,self.rej_small_fea)
            result_pos, nonzero_matrix_pos = self.cur_FM_model(pos_list,pos_list2)
            result_neg, nonzero_matrix_neg = self.cur_FM_model(neg_list,neg_list2)
            diff = (result_pos - result_neg)
            loss = - lsigmoid(diff).sum(dim=0)

            nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)
            nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)
            loss += (reg * nonzero_matrix_pos_).sum(dim=0)
            loss += (reg * nonzero_matrix_neg_).sum(dim=0)

            self.recommender_optimizer.zero_grad()
            loss.backward()
            self.recommender_optimizer.step()
        # end for


    def step(self, action):   #action:0  ask   action:1  recommend   setp=MAX_TURN  done
        done = 0
        print('---------------Turn :{}-------------'.format(self.cur_conver_step))

        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')
            done = 1
        elif action == 0:   #ask large_feature
            print('-->action: ask big features')
            reward, done, acc_feature, rej_feature = self._ask_update()  #update user's profile:  user_acc_feature & user_rej_feature & cand_items
            self._update_cand_items(acc_feature, rej_feature)  # update cand_item and small fea

            if len(acc_feature):   # can reach new large_feature：  update current node and reachable_feature
                self.cur_node_set = acc_feature
                self._updata_reachable_feature(start='large_feature')  # update user's profile: reachable_feature
                #compute feature_score:  fm_score  or fm_score+ent_score

            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_rej_feature))
            print('user_acc_feature:{}'.format(self.user_acc_feature))
            print('user_rej_feature:{}'.format(self.user_rej_feature))

            if self.state_command in [1, 2, 6, 7]:  # update attr_ent
                self._update_feature_entropy()
            if len(self.reachable_feature) != 0:  # if reachable_feature == 0 :cand_item= 1

                if self.ent_way in ['entropy','weight_entropy']:
                    reach_fea_score = self._feature_score()  # compute feature score

                    max_score = max(reach_fea_score)
                    max_ind = reach_fea_score.index(max_score)
                    max_fea_id = self.reachable_feature[max_ind]
                else:#  active_policy

                    al_agent = ALagent(self.user_id,self.target_item,self.user_acc_feature,self.al_Sampler, None, self.kg.ckg_graph,test=True)

                    max_fea_id = al_agent.select_fuzzy_big_sample(self.cur_FM_model,self.reachable_feature,self.user_rej_feature+self.user_acc_feature,self.user_acc_feature)
                # print('max_fea_id:{}'.format(max_fea_id))
                # print('self.reachable_feature:{}'.format(self.reachable_feature))
                max_ind = self.reachable_feature.index(max_fea_id)
                self.reachable_feature.pop(max_ind)
                self.reachable_feature.insert(0, max_fea_id)


        elif action == 1:  #recommend items
            #select topk candidate items to recommend
            cand_item_score = self._item_score()
            item_score_tuple = list(zip(self.cand_items, cand_item_score))
            sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
            self.cand_items, cand_item_score = zip(*sort_tuple)


            #===================== rec update=========
            reward, done = self._recommend_updata()
            #========================================
            if reward == 1:
                print('-->Recommend successfully!')
            else:
                for i in range(self.A.update_fm_count):
                    self.update_FM_model()
                print('----- update current FM {} times -----'.format(self.A.update_fm_count))
                if self.state_command in [1, 2, 6, 7]:  # update attr_ent
                    self._update_feature_entropy()
                print('-->Recommend fail !')


        self.cur_conver_step += 1
        return self._get_state(), reward, done

    def _updata_reachable_feature(self, start='large_feature'):
        # self.reachable_feature = []
        if start == 'user':
            user_like_random_fea = random.choice(bcfg.item_dict[str(self.target_item)]['big_feature_index'])
            self.user_acc_feature.append(user_like_random_fea)  # update user acc_fea
            user_like_random_fea_small = random.choice(bcfg.item_dict[str(self.target_item)]['feature_index'])
            self.acc_small_fea.append(user_like_random_fea_small)  # update user acc_fea
            self.cur_node_set = [user_like_random_fea]

            next_reachable_feature = []
            for cur_node in self.cur_node_set:
                fea_belong_items = bcfg.big_feature_dict[str(cur_node)]['item_index_list']  # A-I

                cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))
                # print('---> A-I-A item length: {}'.format(len(cand_fea_belong_items)))
                for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items]
                    next_reachable_feature.append(bcfg.item_dict[str(item_id)]['big_feature_index'])
                next_reachable_feature = list(set(_flatten(next_reachable_feature)))
            self.reachable_feature = next_reachable_feature  # init reachable_feature

        elif start == 'large_feature':
            next_reachable_feature = []
            for cur_node in self.cur_node_set:
                fea_belong_items = bcfg.big_feature_dict[str(cur_node)]['item_index_list'] # A-I

                cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))
                for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items]
                    next_reachable_feature.append(bcfg.item_dict[str(item_id)]['big_feature_index'])
                next_reachable_feature = list(set(_flatten(next_reachable_feature)))
            self.reachable_feature = next_reachable_feature

    def _feature_score(self):
        reach_fea_score = []
        for feature_id in self.reachable_feature:
            score = self.attr_ent[feature_id]
            reach_fea_score.append(score)
        return reach_fea_score

    def _item_score(self):
        # cand_item_score = []
        # for item_id in self.cand_items:
        #     item_embed = self.ui_embeds[self.user_length + item_id]
        #     score = 0
        #     score += np.inner(np.array(self.user_embed), item_embed)
        #     prefer_embed = self.feature_emb[self.acc_small_fea, :]  #np.array (x*64), small_feature
        #     for i in range(len(self.acc_small_fea)):
        #         score += np.inner(prefer_embed[i], item_embed)
        #     cand_item_score.append(score)
        self.cur_FM_model.eval()
        mini_ui_pair = np.zeros((len(self.cand_items), 2))
        for index, itemID in enumerate(self.cand_items):
            mini_ui_pair[index, :] = [self.user_id, itemID + bcfg.n_users]
        mini_ui_pair = torch.from_numpy(mini_ui_pair).long()
        mini_ui_pair = cuda_(mini_ui_pair)

        static_preference_index = torch.LongTensor(self.acc_small_fea).expand(len(self.cand_items), len(self.acc_small_fea))  # candidate_list, given preference
        static_preference_index = cuda_(static_preference_index)

        static_score, _ = self.cur_FM_model(mini_ui_pair, static_preference_index)
        static_score = static_score.detach().cpu().numpy()

        cand_item_score=static_score.reshape(-1).tolist()
        return cand_item_score


    def _ask_update(self):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0

        feature_groundtrue = bcfg.item_dict[str(self.target_item)]['big_feature_index']

        remove_acced_reachable_fea = self.reachable_feature.copy()   # copy reachable_feature

        acc_feature = list(set(remove_acced_reachable_fea[:1]) & set(feature_groundtrue))
        rej_feature = list(set(remove_acced_reachable_fea[:1]) - set(acc_feature))

        # update user_acc_feature & user_rej_feature
        self.user_acc_feature.append(acc_feature)
        self.user_acc_feature = list(set(_flatten(self.user_acc_feature)))
        self.user_rej_feature.append(rej_feature)
        self.user_rej_feature = list(set(_flatten(self.user_rej_feature)))

        if len(acc_feature):
            reward = self.reward_dict['ask_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']   #update conver_his
        else:
            reward = self.reward_dict['ask_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  #update conver_his

        if self.cand_items == []:  #candidate item set is empty
            done = 1
            reward = self.reward_dict['cand_none']
        return reward, done, acc_feature, rej_feature

    def _update_cand_items(self, acc_feature, rej_feature):

        small_feature_groundtrue = bcfg.item_dict[str(self.target_item)]['feature_index']  # TODO small_ground truth
        if len(acc_feature):    #accept large_feature
            for feature_id in acc_feature:
                feature_small_ids = bcfg.big_feature_dict[str(feature_id)]['small_feature_index_list']
                for small_id in feature_small_ids:
                    if small_id in small_feature_groundtrue:  # user_accept small_tag
                        self.acc_small_fea.append(small_id)
                        feature_items = bcfg.feature_dict[str(small_id)]['item_index_list']
                        self.cand_items = set(self.cand_items) & set(feature_items)   #  itersection
                    # else:  #uesr reject small_tag
                    #     self.rej_small_fea.append(small_id)  #reject no update

            self.cand_items = list(self.cand_items)
        self.cand_len = [len(self.cand_items) >>d & 1 for d in range(self.cand_len_size)][::-1]  # binary


        if len(rej_feature):  # reject large_feature
            for feature_id in rej_feature:
                feature_small_ids = bcfg.big_feature_dict[str(feature_id)]['small_feature_index_list']
                for small_id in feature_small_ids:
                    if small_id not in small_feature_groundtrue:  # user_reject small_tag
                        self.rej_small_fea.append(small_id)

    def _recommend_updata(self):
        print('-->action: recommend items')
        recom_items = self.cand_items[: self.rec_num]    # TOP k item to recommend
        print('recom_items:{}'.format(recom_items))
        user_response=self.user.response(rec_list=recom_items)
        print('user_response：\n {}'.format(user_response))
        # if user_response['accept']:
        if self.target_item in recom_items:
            reward = self.reward_dict['rec_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu'] #update state vector: conver_his
            done = 1
        else:
            reward = self.reward_dict['rec_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']  #update state vector: conver_his
            if len(self.cand_items) > self.rec_num:
                self.cand_items = self.cand_items[self.rec_num:]  #update candidate items
            self.cand_len = [len(self.cand_items) >> d & 1 for d in range(self.cand_len_size)][::-1]  #  binary
            done = 0
        return reward, done


    def _update_feature_entropy(self):
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            #TODO Dataframe
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(bcfg.item_dict[str(item_id)]['feature_index']))
            cand_items_fea_list = list(_flatten(cand_items_fea_list))
            self.attr_count_dict = dict(Counter(cand_items_fea_list))
            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able_large_fea = self.reachable_feature
            for large_fea_id in real_ask_able_large_fea:
                large_ent = 0
                small_feature = bcfg.big_feature_dict[str(large_fea_id)]['small_feature_index_list']
                small_feature_in_cand = list(set(small_feature) & set(self.attr_count_dict.keys()))

                for fea_id in small_feature_in_cand:
                    p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                    p2 = 1.0 - p1
                    if p1 == 1:
                        large_ent += 0
                    else:
                        ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                        large_ent += ent
                self.attr_ent[large_fea_id] =large_ent
        elif self.ent_way == 'weight_entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            cand_item_score = self._item_score()
            cand_item_score_sig = self.sigmoid(cand_item_score)  # sigmoid(score)

            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = bcfg.item_dict[str(item_id)]['feature_index']
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able_large_fea = self.reachable_feature
            sum_score_sig = sum(cand_item_score_sig)
            for large_fea_id in real_ask_able_large_fea:
                large_ent = 0
                small_feature = bcfg.big_feature_dict[str(large_fea_id)]['small_feature_index_list']
                small_feature_in_cand = list(set(small_feature) & set(self.attr_count_dict.keys()))

                for fea_id in small_feature_in_cand:
                    p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig
                    p2 = 1.0 - p1
                    if p1 == 1 or p1 <= 0:
                        large_ent += 0
                    else:
                        ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                        large_ent += ent
                self.attr_ent[large_fea_id] = large_ent

    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()


