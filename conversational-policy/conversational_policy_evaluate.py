import time
from itertools import count
import numpy as np
import torch
import sys
sys.path.append('/home/mengyuan/KGenSam/data-helper')
from data_in import load_pretrain_fm_model,load_pretrain_ns_model,load_pretrain_as_model
from data_out import creat_rl_model_logfile,save_rl_mtric,save_rl_model_log
sys.path.append('/home/mengyuan/KGenSam/configuration')
from base_config import bcfg
sys.path.append('/home/mengyuan/KGenSam/KG')
from knowledge_graph import global_kg
sys.path.append('/home/mengyuan/KGenSam/user-simulator')
from env import BinaryRecommendEnv,EnumeratedRecommendEnv

from utils import cuda_


"""valid"""
def dqn_evaluate(args, kg, agent, filename, i_episode,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler):
    # init valid_env
    if bcfg.data_name=='yelp':
        valid_env = EnumeratedRecommendEnv(kg, args,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler,mode='valid')
    else:
        valid_env = BinaryRecommendEnv(kg, args,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler,mode='valid')

    start = time.time()
    # self.reward_dict = {
    #     'ask_suc': 0.1,
    #     'ask_fail': -0.1,
    #     'rec_suc': 1,
    #     'rec_fail': -0.3,
    #     'until_T': -0.3,  # until MAX_Turn
    #     'cand_none': -0.1
    # }
    # ealuation metric  ST@T
    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    SR_turn_15 = [0]* args.max_turn
    turn_result = []
    result = []
    user_size = valid_env.ui_array.shape[0]
    print('User size in UI_valid: ', user_size)
    valid_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    if bcfg.data_name =='lastfm':
        # valid_size = 4000     # Only do 4000 iteration for the sake of time
        valid_size = 100     # Only do 1000 iteration for the sake of time
        user_size = valid_size
    if bcfg.data_name =='yelp':
        # valid_size = 2500     # Only do 2500 iteration for the sake of time
        valid_size = 50     # Only do 2500 iteration for the sake of time
        user_size = valid_size
    print('The select Validate size : ', valid_size)
    for user_num in range(user_size+1):  #user_size
        # TODO uncommend this line to print the dialog process
        # blockPrint()
        user_id = valid_env.ui_array[valid_env.test_num, 0]
        target_item = valid_env.ui_array[valid_env.test_num, 1]
        valid_env.test_num += 1
        print('\n================valid tuple:{}===================='.format(user_num))
        valid_env.reset_FM_model(pretrain_FM_model)
        state = valid_env.reset_user(user_id,target_item)  # Reset environment and record the starting state
        state = cuda_(torch.unsqueeze(torch.FloatTensor(state), 0))
        for t in count():  # user  dialog
            # action = agent.select_action(state)
            action = agent.policy_net(state).max(1)[1].view(1, 1)
            next_state, reward, done = valid_env.step(action.item())
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
            print('Total evalueation epoch_uesr:{}'.format(user_num + 1))
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
                  mode='valid')
    save_rl_mtric(filename=valid_filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='valid')  # save conversational-policy SR
    print('save valid result successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
    print('success turn:{}'.format(SRturn_all))
    PATH = '{}/Conversational-Policy-log/{}.txt'.format(bcfg.log_root,valid_filename)
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Valid Turn===============\n')
        f.write('Validating {} user tuples\n'.format(user_num))
        for i in range(len(SRturn_all)):
            f.write('Validating SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')


"""test"""
def dqn_test(args, kg, agent, filename, i_episode,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler):
    # init test_env
    if bcfg.data_name=='yelp':
        test_env = EnumeratedRecommendEnv(kg, args,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler,mode='test')
    else:
        test_env = BinaryRecommendEnv(kg, args,pretrain_FM_model,pretrain_neg_sampler,pretrain_al_sampler,mode='test')

    start = time.time()
    # self.reward_dict = {
    #     'ask_suc': 0.1,
    #     'ask_fail': -0.1,
    #     'rec_suc': 1,
    #     'rec_fail': -0.3,
    #     'until_T': -0.3,  # until MAX_Turn
    #     'cand_none': -0.1
    # }
    # ealuation metric  ST@T
    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    SR_turn_15 = [0]* args.max_turn
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    print('User size in UI_test: ', user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    if bcfg.data_name =='lastfm':
        test_size = 4000     # Only do 4000 iteration for the sake of time
        # test_size = 1000     # Only do 1000 iteration for the sake of time
        user_size = test_size
    if bcfg.data_name =='yelp':
        # test_size = 2500     # Only do 2500 iteration for the sake of time
        test_size = 500     # Only do 2500 iteration for the sake of time
        user_size = test_size
    print('The select Test size : ', test_size)
    for user_num in range(user_size+1):  #user_size
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
            print('Total evalueation epoch_uesr:{}'.format(user_num + 1))
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
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(user_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')

