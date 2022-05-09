import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import copy
import math
import networkx as nx
from sklearn.metrics import roc_auc_score

import random

from time import time
import numpy as np
import argparse

import sys
sys.path.append('/home/mengyuan/AUM-V4/data-helper')
from data_loader import KG_Data_loader,FM_Data_loader
from data_in import load_pretrain_fm_model

sys.path.append('/home/mengyuan/AUM-V4/FM')
from factorization_machine import FM
from fm_attribute_evaluate import evaluate_fm_attribute_reward_for_ASampler


sys.path.append('/home/mengyuan/AUM-V4/configuration')
from base_config import bcfg

sys.path.append('/home/mengyuan/AUM-V4/active-sampler')
from rewardshaper import RewardShaper
from al_policy_evaluate import degprocess,localdiversity,entropy,mean_std

sys.path.append('/home/mengyuan/AUM-V4')
from utils import cuda_




class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, batch_size, bias=False):

        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):

        support = torch.einsum("jik,kp->jip", input, self.weight)
        if self.bias is not None:
            support = support + self.bias
        support = torch.reshape(support, [support.size(0), -1])
        output = torch.spmm(adj, support)
        output = torch.reshape(output, [output.size(0), self.batch_size, -1])
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# vanilla GCN
class ALPolicy2(nn.Module):

    def __init__(self, args):

        super(ALPolicy2, self).__init__()
        self.args = args
        self.statedim = 0
        if self.args.use_entropy:
            self.statedim += 1
        if self.args.use_degree:
            self.statedim += 1
        if self.args.use_local_diversity:
            self.statedim += 2
        if self.args.use_select:
            self.statedim += 1
        self.pnhid=args.pnhid
        self.batch_size=args.batch_size
        self.gcn = nn.ModuleList()
        for i in range(len(self.pnhid)):
            if (i == 0):
                self.gcn.append(GraphConvolution(self.statedim, self.pnhid[i], self.batch_size, bias=True))
            else:
                self.gcn.append(GraphConvolution(self.pnhid[i - 1], self.pnhid[i], self.batch_size, bias=True))

        self.output_layer = nn.Linear(self.pnhid[-1], 1, bias=False)

    def forward(self, state, adj):

        x = state.transpose(0, 1)
        for layer in self.gcn:
            x = F.relu(layer(x, adj))
        x = self.output_layer(x).squeeze(-1).transpose(0, 1)
        return x

class ALPolicy(nn.Module):

    def __init__(self, args):
        super(ALPolicy,self).__init__()
        self.args = args
        self.statedim=0
        if self.args.use_entropy:
            self.statedim+=1
        if self.args.use_degree:
            self.statedim+=1
        if self.args.use_local_diversity:
            self.statedim+=2
        if self.args.use_select:
            self.statedim+=1
        self.lin1 = nn.Linear(self.statedim,args.pnhid[0])
        self.lin2 = nn.Linear(args.pnhid[0],args.pnhid[0])
        self.lin3 = nn.Linear(args.pnhid[0], 1)
        stdv = 1. / math.sqrt(self.lin1.weight.size(1))

        self.lin1.weight.data.uniform_(-stdv, stdv)
        self.lin2.weight.data.uniform_(-stdv, stdv)
        self.lin3.weight.data.uniform_(-stdv, stdv)

        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.zeros_(self.lin3.bias)


    def forward(self,state,adj):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        logits = self.lin3(x).squeeze(-1)
        return logits


class ALagent():
    def __init__(self,user_id,item_id,preference_list,AL_model,al_optimizer,graph,test=False):
        self.test=test

        self.user_id=user_id
        self.item_id=item_id
        self.preference_list=preference_list

        self.args = bcfg.get_Active_Sampler_parser()
        self.rshaper = RewardShaper(self.args)
        self.reset_FM_model()

        self.policy = AL_model
        self.al_optimizer = al_optimizer

        self.maxturns=self.args.mt
        self.batch_size=self.args.batch_size
        self.FEATURE_POOLS = bcfg.attribute_list

        self.graph = graph

        # self.adj = nx.adjacency_matrix(self.graph).astype(np.float32)
        # self.deg = torch.sparse.sum(self.adj, dim=1).to_dense()
        # self.normdeg = self.deg / self.deg.max()

        self.adj = None
        self.deg = None
        self.normdeg = None

        self.epoch_loss=0

        self.lsigmoid = nn.LogSigmoid()

        self.valid_fm_pickle_file =self._initialize_valid_fm_data()
        self.valid_fm_index=self._get_valid_index(self.valid_fm_pickle_file)

        self.allnodes_output=None

    def _initialize_valid_fm_data(self):
        if self.test:
            return FM_Data_loader(mode='test')
        else:
            return FM_Data_loader(mode='valid')

    def _get_valid_index(self,pickle_file):
        # pickle_file_length = len(pickle_file[0])
        bs = 32
        le_ri = []
        sd = 0
        while len(le_ri) < bs:
            random.seed(sd)
            tp = random.randint(0, len(pickle_file[2]) - 1)
            # print(len(pickle_file[2]),tp)
            i_neg2_output = pickle_file[2][tp]
            if tp not in le_ri and i_neg2_output is not None and len(i_neg2_output) != 0:
                le_ri.append(tp)
            sd += 1
        return le_ri


    def _get_allnodes_output(self, given_preference):

        def fm_predict_attributes(given_preference,to_test):
            # to_test = self.FEATURE_POOLS
            gp = self.FM_model.attri_emb(torch.LongTensor(given_preference).cuda())[..., :].cpu().detach().numpy()
            emb_weight = self.FM_model.attri_emb.weight[..., :].cpu().detach().numpy()
            result = list()

            for test_feature in to_test:
                temp = 0
                for i in range(gp.shape[0]):
                    temp += np.inner(gp[i], emb_weight[test_feature])
                result.append(temp)
            return result

        output = np.array(fm_predict_attributes(given_preference,to_test = self.FEATURE_POOLS))
        # predictions: (33,)
        output = torch.tensor(output.reshape(1,1,len(self.FEATURE_POOLS)),dtype=torch.float)
        # print('allnodes_output:{}'.format(output))
        return output.detach()

    def _update_FM_model(self,turn):
        self.FM_model.train()
        # PAD_IDX1=bcfg.PAD_IDX1
        # PAD_IDX2=bcfg.PAD_IDX2
        pos_list=[]
        preference_list=[]
        residual_feature, neg_feature = [], []

        this_residual_feature=[i for i in self.preference_list if i not in self.given_preference]
        remain_feature = [i for i in self.FEATURE_POOLS if i not in self.given_preference]
        this_neg_feature = np.random.choice(remain_feature, len(this_residual_feature))

        residual_feature.append(torch.LongTensor(this_residual_feature))
        neg_feature.append(torch.LongTensor(this_neg_feature))
        preference_list.append(torch.LongTensor(self.preference_list))
        pos_list.append(torch.LongTensor([self.user_id, self.item_id + bcfg.n_users]))

        pos_list = pad_sequence(pos_list, batch_first=True, padding_value=bcfg.PAD_IDX1)
        residual_feature = pad_sequence(residual_feature, batch_first=True, padding_value=bcfg.PAD_IDX2)
        neg_feature = pad_sequence(neg_feature, batch_first=True, padding_value=bcfg.PAD_IDX2)
        preference_list = pad_sequence(preference_list, batch_first=True, padding_value=bcfg.PAD_IDX2)

        preference_list,pos_list,residual_feature,neg_feature=cuda_(preference_list),cuda_(pos_list),cuda_(residual_feature),cuda_(neg_feature)

        A = self.FM_model.attri_emb(preference_list)[..., :]
        user_emb = self.FM_model.ui_emb(pos_list[:, 0])[..., :].unsqueeze(dim=1).detach()
        A = torch.cat([A, user_emb], dim=1)
        B = self.FM_model.attri_emb(residual_feature)[..., :]
        C = self.FM_model.attri_emb(neg_feature)[..., :]

        D = torch.matmul(A, B.transpose(2, 1))
        E = torch.matmul(A, C.transpose(2, 1))

        p_vs_residual = D.view(D.shape[0], -1, 1)
        p_vs_neg = E.view(E.shape[0], -1, 1)

        p_vs_residual = p_vs_residual.sum(dim=1)
        p_vs_neg = p_vs_neg.sum(dim=1)
        diff = (p_vs_residual - p_vs_neg)
        temp = - self.lsigmoid(diff).sum(dim=0)
        loss = temp
        self.epoch_loss += temp.data

        #print('turn:{} preference grad norm: {}'.format(turn, torch.norm(self.FM_model.attri_emb.weight.grad)))
        print('Turn {} fm loss is: {} (diff:{})'.format(turn,self.epoch_loss,temp.data))

        self.fm_optimizer_attri.zero_grad()
        loss.backward()
        self.fm_optimizer_attri.step()

    def _make_state(self, probs):
        selected=torch.tensor(self.given_preference)
        # probs.size: torch.Size([10, 2708, 7])
        # entro.size: torch.Size([10, 2708])
        # state.size: torch.Size([10, 2708, 5])
        # print('probs.size:{}'.format(probs.size()))
        # print('probs:{}'.format(probs))
        entro = entropy(probs)
        # entro = normalizeEntropy(entro, probs.size(-1))  ## in order to transfer
        # print('entro.size:{}'.format(entro.size()))
        # probs.size: torch.Size([1, 33, 1])
        # entro.size: torch.Size([1, 33])
        features = []
        if self.args.use_entropy:
            features.append(entro)
        if self.args.use_degree:
            deg = degprocess(self.deg.expand([probs.size(0)] + list(self.deg.size())))
            features.append(deg)
        if self.args.use_local_diversity:
            mean_kl_ht, mean_kl_th = localdiversity(probs, self.adj, self.deg)
            features.extend([mean_kl_ht, mean_kl_th])
        if self.args.use_select:
            features.append(selected)
        # print('features:{}'.format(features))
        state = torch.stack(features, dim=-1)
        return state


    def get_reward(self,turn):
        if not self.test:
            self.allnodes_output=self._get_allnodes_output(self.given_preference)
            self._update_FM_model(turn)
        else:
            self.given_preference=[]
            self.allnodes_output=self._get_allnodes_output(self.given_preference)
        ##########################################
        return evaluate_fm_attribute_reward_for_ASampler(self.FM_model,turn,self.valid_fm_pickle_file,self.valid_fm_index)


    def reset_FM_model(self):
        self.FM_model = FM(args_config=bcfg.get_FM_parser(), cfg=bcfg)
        if self.args.pretrain_fm:
            # pretrain_r defaut=true
            # 一般都是在预训练过的fm上训练neg_sampler
            model_dict = load_pretrain_fm_model()
            self.FM_model.load_state_dict(model_dict)
        self.FM_model=cuda_(self.FM_model)

        param_bias_attri = list()
        i = 0
        for name, param in self.FM_model.named_parameters():
            # print(name, param)
            if i in [0, 2]:
                param_bias_attri.append(param)
            i += 1
        self.fm_optimizer_attri = torch.optim.Adam(param_bias_attri, lr=self.args.rllr)


    def get_state(self):
        output =self.allnodes_output.transpose(1, 2)
        state = self._make_state(output)
        return state

    def get_pool(self):
        pool=[]
        for fea in self.FEATURE_POOLS:
            if fea not in self.actions:
                pool.append(fea)
        # print('actions:{}'.format(self.actions))
        # print('given_preference:{}'.format(self.given_preference))
        # print('preference_list:{}'.format(self.preference_list))
        # print('pool:{}'.format(pool))
        return pool

    def select_actions(self,logits,pool):
        # print('logits: {}'.format(logits))
        # print('logits.shape: {}'.format(logits.shape)) #torch.Size([1, 33])
        prob = F.softmax(logits,dim=1)
        c = Categorical(prob)
        # print('prob: {}'.format(prob))
        pred_data = logits.cpu().detach().numpy().data.tolist()[0]
        # print('pred_data: {}'.format(pred_data))
        sorted_index = sorted(range(len(pred_data)), key=lambda k: pred_data[k], reverse=True)
        unasked_max = None
        for item in sorted_index:
            if item in pool:
                unasked_max = item
                break
        action = Variable(torch.IntTensor([unasked_max]))  # make it compatible with torch
        # print('Take action : {}'.format(unasked_max))
        logprob = c.log_prob(action.cuda())

        self.valid_probs=[]
        for item in pool:
            self.valid_probs.append(pred_data[item])

        return unasked_max,action, logprob, prob

    def playOneEpisode(self, episode):
        print('####################### 开始episode {} ######################### '.format(episode))
        rewards, logp_actions, p_actions = [], [], []
        self.states, self.pools = [], []
        self.actions, self.given_preference=[], []
        initialrewards = self.get_reward(0)
        rewards.append(initialrewards)
        self.entropy_reg = []
        # self.action_index = np.zeros([self.args.batch_size, self.maxturns])
        start = time()
        for turn in range(1,self.maxturns+1):
            print('------------------------- turn {} ------------------------- '.format(turn))
            state = self.get_state()
            self.states.append(state)
            pool = self.get_pool()
            # print('Pool : {}'.format(str(pool)))
            self.pools.append(pool)

            logits = self.policy(state.cuda(), self.adj)
            # logits.shape: torch.Size([10, 2708])
            action_feature,action, logp_action, p_action = self.select_actions(logits, pool)

            # self.action_index[:, turn-1] = action.detach().cpu().numpy()
            print('Take action : {}'.format(action_feature))
            self.actions.append(action_feature)
            if action_feature in self.preference_list:
                self.given_preference.append(action_feature)

            logp_actions.append(logp_action)
            p_actions.append(p_action)

            reward=self.get_reward(turn)
            print('Reward : {}'.format(sum(reward)))
            rewards.append(reward)
            # self.entropy_reg.append(
            #     -(self.valid_probs * torch.log(1e-6 + self.valid_probs)).sum(dim=1) / np.log(self.valid_probs.size(1)))

        print('Episode {} takes {} seconds'.format(episode, time() - start))
        logp_actions = torch.stack(logp_actions)
        p_actions = torch.stack(p_actions)
        # self.entropy_reg = torch.stack(self.entropy_reg).cuda()
        finalrewards = self.get_reward(self.maxturns)

        self.cur_rewards=finalrewards
        # print('finalrewards:{}'.format(finalrewards))

        micfinal, _ = mean_std(finalrewards)
        print("Episode {} MEAN REWARD in validation is {}".format(episode, micfinal))

        shapedrewards = self.rshaper.reshape(rewards, finalrewards, logp_actions.detach().cpu().numpy())
        return shapedrewards, logp_actions, p_actions

    def finishEpisode(self, rewards, logp_actions, p_actions):

        rewards = torch.from_numpy(rewards).cuda().type(torch.float32)

        # losses = logp_actions * rewards + self.args.entcoef * self.entropy_reg
        losses = logp_actions * rewards
        # print('rewards.size:{}'.format(rewards.size()))
        # print('logp_actions.size:{}'.format(logp_actions.size()))
        # print('losses.size:{}'.format(losses.size()))

        # rewards.size: torch.Size([15, 5437])
        # logp_actions.size: torch.Size([15, 1])  [15, 5437?]
        # losses.size: torch.Size([15, 5437])

        loss = -torch.mean(torch.sum(losses, dim=0))
        self.al_optimizer.zero_grad()
        loss.backward()
        self.al_optimizer.step()

        return loss.item()

    def select_fuzzy_sample(self,current_FM_model,can_feature_pool,asked_feature,known_feature):
        # asked_feature list() of str(cate id)
        self.FM_model=current_FM_model
        self.actions= asked_feature
        self.given_preference= known_feature
        self.allnodes_output=self._get_allnodes_output(known_feature)
        state=self.get_state()
        # pool = self.get_pool()
        pool = can_feature_pool


        logits = self.policy(state.cuda(), self.adj)
        action_feature, action, logp_action, p_action = self.select_actions(logits, pool)
        prob = F.softmax(logits,dim=1)
        c = Categorical(prob)
        pred_data = logits.cpu().detach().numpy().data.tolist()[0]
        sorted_index = sorted(range(len(pred_data)), key=lambda k: pred_data[k], reverse=True)
        action_feature = None
        for item in sorted_index:
            if item in pool:
                action_feature = item
                break
        return action_feature

    def get_big_feature_score(self,logits,pool):
        big_feature_score_dict=dict()
        prob = F.softmax(logits,dim=1)
        c = Categorical(prob)
        # print('prob: {}'.format(prob))
        pred_data = logits.cpu().detach().numpy().data.tolist()[0]
        # print('pred_data: {}'.format(pred_data))
        for small_f_index in range(len(pred_data)):
            if small_f_index in pool and str(small_f_index) in bcfg.feature_dict.keys():
                big_f=str(bcfg.feature_dict[str(small_f_index)]['big_feature_index']) #int to str
                if big_f not in big_feature_score_dict.keys():
                    big_feature_score_dict[big_f]=pred_data[small_f_index]
                else:
                    big_feature_score_dict[big_f]+=pred_data[small_f_index]
        return big_feature_score_dict

    def select_fuzzy_big_sample(self,current_FM_model,can_feature_pool,asked_feature,known_feature):
        # asked_feature list() of str(cate id)

        self.FM_model=current_FM_model

        self.actions=[]
        for big_f in asked_feature:
            self.actions.extend(bcfg.big_feature_dict[str(big_f)]['small_feature_index_list'])
        self.actions=list(set(self.actions))

        self.given_preference=[]
        for big_f in known_feature:
            self.given_preference.extend(bcfg.big_feature_dict[str(big_f)]['small_feature_index_list'])
        self.given_preference=list(set(self.given_preference))

        self.allnodes_output=self._get_allnodes_output(self.given_preference)
        state=self.get_state()

        pool=[]
        for big_f in can_feature_pool:
            pool.extend(bcfg.big_feature_dict[str(big_f)]['small_feature_index_list'])
        pool=list(set(pool))

        logits = self.policy(state.cuda(), self.adj)
        big_feature_score_dict = self.get_big_feature_score(logits, pool)

        sorted_big_feature_score = sorted(big_feature_score_dict.items(), key=lambda x: x[1], reverse=True)
        action_feature = None
        for bf,bf_score in sorted_big_feature_score:
            if int(bf) in can_feature_pool:
                action_feature = int(bf)
                break
        return action_feature