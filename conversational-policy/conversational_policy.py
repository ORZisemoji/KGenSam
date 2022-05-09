
import warnings
warnings.filterwarnings("ignore")
import math
import random
import numpy as np
import os
import sys
sys.path.append('/home/mengyuan/KGenSam/data-hepler')
from data_in import load_rl_model
from data_out import save_rl_model
# sys.path.append('..')
from utils import cuda_
#TODO select env
sys.path.append('/home/mengyuan/KGenSam/user-simulator')
from env import BinaryRecommendEnv,EnumeratedRecommendEnv
sys.path.append('/home/mengyuan/KGenSam/conversational-policy')
from conversational_policy_evaluate import dqn_evaluate

from collections import namedtuple
import argparse
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
import time

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_space, hidden_size, action_space):
            super(DQN, self).__init__()
            self.state_space = state_space
            self.action_space = action_space
            self.fc1 = nn.Linear(self.state_space, hidden_size)
            self.fc1.weight.data.normal_(0, 0.1)   # initialization
            self.out = nn.Linear(hidden_size, self.action_space)
            self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class Agent(object):
    def __init__(self, memory, state_space, hidden_size, action_space, EPS_START = 0.9, EPS_END = 0.05, EPS_DECAY = 200):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.policy_net = cuda_(DQN(state_space, hidden_size, action_space))
        self.target_net = cuda_(DQN(state_space, hidden_size, action_space))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = memory


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return cuda_(torch.tensor([[random.randrange(2)]], dtype=torch.long))

    def optimize_model(self, BATCH_SIZE, GAMMA):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.uint8)
        n_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(n_states)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = cuda_(torch.zeros(BATCH_SIZE))
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch


        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.data

    
    def save_policy_model(self, epoch):
        save_rl_model(model=self.policy_net, epoch=epoch)

    def load_policy_model(self, epoch):
        model_dict = load_rl_model(epoch=epoch)
        self.policy_net.load_state_dict(model_dict)



