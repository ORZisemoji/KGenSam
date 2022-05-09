
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
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



def train(kg,args,filename):

    # load pretrain fm
    pretrain_FM_model=FM(args_config=bcfg.get_FM_parser(),cfg=bcfg)
    pretrain_FM_model_dict=load_pretrain_fm_model()
    pretrain_FM_model.load_state_dict(pretrain_FM_model_dict)
    cuda_(pretrain_FM_model)

    # load_pretrain_ns_model
    # data_params={
    #         "n_users": global_kg.n_users,
    #         "n_items": global_kg.n_items,
    #         "n_relations": global_kg.n_relations + 2,
    #         "n_entities": global_kg.n_entities,
    #         "n_nodes": global_kg.n_items,
    #         "item_range": global_kg.item_range,
    #     }
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
        #epoch-40-Negative_Sampler_model-k_step-2
        pretrain_neg_sampler_dict=load_pretrain_ns_model(filename='epoch-40-Negative_Sampler_model-k_step-2')
        # pretrain_neg_sampler_dict=load_pretrain_ns_model(filename='Negative_Sampler_model-k_step-2')
    else:# LASTFM
        pretrain_neg_sampler_dict=load_pretrain_ns_model(filename='Negative_Sampler_model-k_step-2')
    pretrain_neg_sampler.load_state_dict(pretrain_neg_sampler_dict)
    cuda_(pretrain_neg_sampler)

    # load_pretrain_as_model
    pretrain_al_sampler=ALPolicy(args=bcfg.get_Active_Sampler_parser())
    pretrain_al_sampler_dict=load_pretrain_as_model()
    pretrain_al_sampler.load_state_dict(pretrain_al_sampler_dict)
    cuda_(pretrain_al_sampler)


    # init env
    if bcfg.data_name=='yelp':
        env = EnumeratedRecommendEnv(kg, args,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler,mode=args.mode)
    else:
        env = BinaryRecommendEnv(kg, args,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler,mode=args.mode)

    state_space = env.state_space
    action_space = env.action_space
    memory = ReplayMemory(args.memory_size) #10000
    agent = Agent(memory=memory, state_space=state_space, hidden_size=args.hidden, action_space=action_space)
    tt = time.time()
    # self.reward_dict = {
    #     'ask_suc': 0.1,
    #     'ask_fail': -0.1,
    #     'rec_suc': 1,
    #     'rec_fail': -0.3,
    #     'until_T': -0.3,  # until MAX_Turn
    #     'cand_none': -0.1
    # }
    #ealuation metric  ST@T
    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    loss = cuda_(torch.tensor(0, dtype=torch.float))
    start = time.time()
    #agent load policy parameters
    if args.pretrain_rl_epoch != 0 :
        print('Staring loading rl model in epoch {}'.format(args.pretrain_rl_epoch))
        agent.load_policy_model(epoch=args.pretrain_rl_epoch)

    for i_episode in range(args.pretrain_rl_epoch+1, args.max_epoch++1): #args.max_epoch
        if args.mode == 'train':
            users = list(env.user_weight_dict.keys())
            # self.user_id = np.random.choice(users, p=list(self.user_weight_dict.values())) # select user  according to user weights
            user_id = np.random.choice(users)
            target_item = np.random.choice(env.ui_dict[str(user_id)])
        elif args.mode == 'test':
            user_id = env.ui_array[env.test_num, 0]
            target_item = env.ui_array[env.test_num, 1]
            env.test_num += 1
        # blockPrint()  # Block user-agent process output
        print('\n================new tuple:{}===================='.format(i_episode))
        # Reset environment and record the starting state
        env.reset_FM_model(pretrain_FM_model)
        state = env.reset_user(user_id,target_item)
        state = cuda_(torch.unsqueeze(torch.FloatTensor(state), 0))
        for t in count():   # user  dialog
            action = agent.select_action(state)
            next_state, reward, done = env.step(action.item())
            next_state = cuda_(torch.tensor([next_state], dtype=torch.float))
            reward = cuda_(torch.tensor([reward], dtype=torch.float))

            if done:
                next_state = None
            agent.memory.push(state, action, next_state, reward)
            state = next_state

            newloss = agent.optimize_model(args.batch_size, args.gamma)
            if newloss is not None:
                loss += newloss

            if done:
                if reward.item() == 1:  #recommend successfully
                    if t < 5:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                    elif t < 10:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1
                AvgT += t
                break
        if i_episode % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # enablePrint() # Enable print function

        if i_episode % args.observe_num == 0 and i_episode > 0:
            print('loss : {} in episode {}'.format(loss.item()/args.observe_num, i_episode))
            if i_episode % (args.observe_num * 2) == 0 and i_episode > 0:
                print('save model in episode {}'.format(i_episode))
                save_rl_model_log(filename=filename, epoch=i_episode, epoch_loss=loss.item()/args.observe_num, train_len=args.observe_num)
                SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT/args.observe_num]
                save_rl_mtric(filename=filename, epoch=i_episode, SR=SR, spend_time=time.time()-tt,mode=args.mode)  #save RL metric

            if i_episode % (args.observe_num * 4) == 0 and i_episode > 0:
                agent.save_policy_model(epoch=i_episode) # save RL policy model
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{} Total epoch_uesr:{}'.format(SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT/args.observe_num, i_episode+1))
            print('spend time: {}'.format(time.time()-start))
            SR5, SR10, SR15, AvgT = 0, 0, 0, 0
            loss = cuda_(torch.tensor(0, dtype=torch.float))
            tt = time.time()

        if i_episode % (args.observe_num * 4) == 0 and i_episode > 0:
            print('Evaluating on Test tuples!')
            dqn_evaluate(args, kg, agent, filename, i_episode,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler)


if __name__ == '__main__':
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
    parser.add_argument('--negSampler', type=int, default=bcfg.negSampler, help='use negative sampler or not')
    parser.add_argument('--alSampler', type=int, default=bcfg.alSampler, help='use active sampler or not')

    parser.add_argument('--state_command', type=int, default=7, help='select state vector')
    parser.add_argument('--observe_num', type=int, default=200, help='the number of epochs to save RL model and metric')
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


    # """fix the random seed"""
    # A.seed = 2021
    # random.seed(A.seed)
    # np.random.seed(A.seed)
    # torch.manual_seed(A.seed)
    # torch.cuda.manual_seed(A.seed)

    # with two samplers:
    #     A.update_fm_count=4
    #     A.entropy_method='active_policy'


    if A.alSampler:
        A.entropy_method='active_policy'
    else:
        A.entropy_method='entropy'

    if bcfg.data_name=='yelp': # too slow
        A.observe_num=50
    else:
        A.observe_num=100

    print('attr_num: {}'.format(bcfg.n_attributes))
    print('entropy_method: {}'.format(A.entropy_method))
    filename = 'v4-Conversational-Policy-mode-{}-negSampler-{}-state_command-{}-entropy_method-{}-reward0{}'.format(
    A.mode,A.negSampler,A.state_command,A.entropy_method,A.reward_option)
    # creat new log file
    if not A.pretrain_rl_epoch:
        creat_rl_model_logfile(filename)

    bcfg.filename=filename

    train(kg=global_kg,args=A,filename=filename)