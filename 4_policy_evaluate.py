import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
sys.path.append('/home/mengyuan/KGenSam')
from utils import cuda_,set_random_seed
set_random_seed()

import warnings
warnings.filterwarnings('ignore')



sys.path.append('/home/mengyuan/KGenSam/FM')
from factorization_machine import FM
sys.path.append('/home/mengyuan/KGenSam/negative-sampler')
from neg_policy import KGPolicy
sys.path.append('/home/mengyuan/KGenSam/active-sampler')
from al_policy import ALPolicy
sys.path.append('/home/mengyuan/KGenSam/data-helper')
from data_in import load_pretrain_fm_model,load_pretrain_ns_model,load_pretrain_as_model
from data_out import creat_rl_model_logfile,save_rl_mtric,save_rl_model_log
sys.path.append('/home/mengyuan/KGenSam/configuration')
from base_config import bcfg
sys.path.append('/home/mengyuan/KGenSam/KG')
from knowledge_graph import global_kg
sys.path.append('/home/mengyuan/KGenSam/conversational-policy')
from conversational_policy import ReplayMemory,Agent
from conversational_policy_evaluate import dqn_evaluate
sys.path.append('/home/mengyuan/KGenSam/user-simulator')
from env import BinaryRecommendEnv,EnumeratedRecommendEnv


import pickle
import torch
import argparse
from itertools import count
import time
import numpy as np

import copy
import random
import json

def get_policy_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=2021, help='random seed.')
    parser.add_argument('--max_epoch', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--target_update', type=int, default=20, help='the number of epochs to update policy parameters')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument(
        "-rlr", type=float, default=0.001, dest='rlr', help="Learning rate recommender."
    )
    parser.add_argument('--hidden', type=int, default=512, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')

    parser.add_argument('--entropy_method', type=str, default='entropy', help='entropy_method is one of [active_policy,entropy, weight_entropy]')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--cand_len_size', type=int, default=20, help='binary state size for the length of candidate items')

    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--negSampler', type=int, default=1, help='use negative sampler or not')
    parser.add_argument('--alSampler', type=int, default=1, help='use active sampler or not')

    parser.add_argument('--state_command', type=int, default=7, help='select state vector')
    parser.add_argument('--observe_num', type=int, default=10, help='the number of epochs to save RL model and metric')
    parser.add_argument('--pretrain_rl_epoch', type=int, default=0, help='the epoch of loading RL model')
    parser.add_argument('--update_fm_count', default=4, type=int, dest='update_fm_count', help='update_fm_count')  # 训练时设为0，评估时设为4
    # how many times to do reflection
    '''
    # conver_his: Conversation_history;   attr_ent: Entropy of attribute ; cand_len: the length of candidate item set 
    # state_command:1   self.user_embed, self.conver_his, self.attr_ent, self.cand_len
    # state_command:2   self.attr_ent
    # state_command:3   self.conver_his
    # state_command:4   self.cond_len
    # state_command:5   self.user_embedding
    # state_command:6   self.conver_his, self.attr_ent, self.cand_len
    # state_command:7   self.conver_his, self.cand_len
    '''
    parser.add_argument('--reward_option', type=int, default=0, help='choose reward dict [0,1,2,3,11]')  # 0 maen origin scpr reward dict

    """initialize args"""
    A = parser.parse_args()

    return A

def policy_evaluate(args, kg, filename, i_episode,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler):
    # init test_env
    if bcfg.data_name=='yelp':
        test_env = EnumeratedRecommendEnv(kg, args,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler,mode='test')
    else:
        test_env = BinaryRecommendEnv(kg, args,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler,mode='test')

    start = time.time()

    # ealuation metric  ST@T
    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    SR_turn_15 = [0]* args.max_turn
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    print('User size in UI_test: ', user_size)
    test_filename = 'TEST-epoch-{}-'.format(i_episode) + filename

    test_size = user_size
    print('The select TEST size : ', test_size)

    """agent"""
    state_space = test_env.state_space
    action_space = test_env.action_space
    memory = ReplayMemory(args.memory_size)  # 10000
    agent = Agent(memory=memory, state_space=state_space, hidden_size=args.hidden, action_space=action_space)
    agent.load_policy_model(epoch=i_episode)


    for user_num in range(user_size):  #user_size
        # TODO uncommend this line to print the dialog process
        # blockPrint()
        user_id = test_env.ui_array[test_env.test_num, 0]
        target_item = test_env.ui_array[test_env.test_num, 1]
        test_env.test_num += 1
        print('\n================test tuple:{}===================='.format(user_num))
        test_env.reset_FM_model(pretrain_FM_model)
        state = test_env.reset_user(user_id,target_item)  # Reset environment and record the starting state
        state = cuda_(torch.unsqueeze(torch.FloatTensor(state), 0))
        for t in count():  # user  dialog
            # action = agent.select_action(state)
            action = agent.policy_net(state).max(1)[1].view(1, 1)
            next_state, reward, done = test_env.step(action.item())
            next_state = cuda_(torch.tensor([next_state], dtype=torch.float))
            reward = cuda_(torch.tensor([reward], dtype=torch.float))

            if done:
                next_state = None
            state = next_state
            if done:
                if reward.item() == 1:  # recommend successfully
                    SR_turn_15 = [v+1 if i>t  else v for i, v in enumerate(SR_turn_15) ]
                    if t < 5:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                    elif t < 10:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1

                AvgT += t+1
                break
        # enablePrint()
        if user_num % args.observe_num == 0 and user_num > 0:
            SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT / args.observe_num]
            SR_TURN = [i/args.observe_num for i in SR_turn_15]
            print('Total test epoch_uesr:{}'.format(user_num + 1))
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                       float(user_num) * 100 / user_size))
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num,
                                                AvgT / args.observe_num, user_num + 1))
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5, SR10, SR15, AvgT = 0, 0, 0, 0
            SR_turn_15 = [0] * args.max_turn


    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean]
    save_rl_mtric(filename=filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test')
    save_rl_mtric(filename=test_filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test')  # save conversational-policy SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
    print('success turn:{}'.format(SRturn_all))
    PATH = '{}/Conversational-Policy-log/{}.txt'.format(bcfg.log_root,test_filename)
    with open(PATH, 'a') as f:
        f.write('Training epoch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(user_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')

if __name__ == '__main__':

    A=get_policy_parser()
    if bcfg.data_name == 'yelp':  # too slow
        A.update_fm_count = 40

    kg=global_kg
    filename = 'Test4-Conversational-Policy-mode-{}-update_fm_count-{}-state_command-{}-entropy_method-{}-reward0{}'.format(
        A.mode, A.update_fm_count, A.state_command, A.entropy_method, A.reward_option)

    # load pretrain fm
    pretrain_FM_model=FM(args_config=bcfg.get_FM_parser(),cfg=bcfg)
    pretrain_FM_model_dict=load_pretrain_fm_model()
    pretrain_FM_model.load_state_dict(pretrain_FM_model_dict)
    cuda_(pretrain_FM_model)

    # load_pretrain_ns_model
    data_params={
        "n_users": global_kg.n_users,
        "n_items": global_kg.n_items,
        "n_relations": global_kg.n_relations + 2,
        "n_entities": global_kg.n_entities,
        "n_nodes": global_kg.entity_range[1] + 1,
        "item_range": global_kg.item_range,
    }
    pretrain_neg_sampler=KGPolicy(rec=pretrain_FM_model, params=bcfg.get_Negative_Sampler_parser(),data_params=data_params)
    # {'n_users': 1801,
    #  'n_items': 7432,
    #  'n_relations': 4,
    #  'n_entities': 7465,
    #  'n_nodes': 9266,
    #  'item_range': (1801, 9231)}
    if bcfg.data_name=='yelp':
        # YELP
        pretrain_neg_sampler_dict=load_pretrain_ns_model(filename='epoch-60-Negative_Sampler_model-k_step-2')
        i_episode=5000
    else:# LASTFM
        pretrain_neg_sampler_dict=load_pretrain_ns_model(filename='Negative_Sampler_model-k_step-2')
        i_episode = 50000
    filename = 'CRSmodel-{}-'.format(i_episode)+filename
    creat_rl_model_logfile(filename)

    pretrain_neg_sampler.load_state_dict(pretrain_neg_sampler_dict)
    cuda_(pretrain_neg_sampler)

    # load_pretrain_as_model
    pretrain_al_sampler=ALPolicy(args=bcfg.get_Active_Sampler_parser())
    pretrain_al_sampler_dict=load_pretrain_as_model()
    pretrain_al_sampler.load_state_dict(pretrain_al_sampler_dict)
    cuda_(pretrain_al_sampler)

    args = A
    policy_evaluate(args, kg, filename, i_episode, pretrain_FM_model, pretrain_neg_sampler, pretrain_al_sampler)
