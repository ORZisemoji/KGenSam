
from torch.utils.data import DataLoader
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import scipy.sparse as sp
import pickle

from time import time
sys.path.append('/home/mengyuan/AUM-V4/KG')
sys.path.append('/home/mengyuan/AUM-V4/data-helper')
sys.path.append('/home/mengyuan/AUM-V4/configuration')
from base_config import bcfg

class KGTrainGenerator(Dataset):
    def __init__(self, graph):
        self.graph = graph

        self.user_dict = graph.train_user_dict
        self.exist_users = list(graph.exist_users)
        self.low_item_index = graph.item_range[0]
        self.high_item_index = graph.item_range[1]

    def __len__(self):
        return self.graph.n_train

    def __getitem__(self, index):
        out_dict = {}

        user_dict = self.user_dict
        # randomly select one user.
        u_id = random.sample(self.exist_users, 1)[0]
        out_dict["u_id"] = u_id

        # randomly select one positive item.
        pos_items = user_dict[u_id]
        n_pos_items = len(user_dict[u_id])

        pos_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
        pos_i_id = pos_items[pos_idx]

        out_dict["pos_i_id"] = pos_i_id

        neg_i_id = self.get_random_neg(pos_items, [])
        out_dict["neg_i_id"] = neg_i_id

        return out_dict

    def get_random_neg(self, pos_items, selected_items):
        while True:
            neg_i_id = np.random.randint(
                low=self.low_item_index, high=self.high_item_index, size=1
            )[0]

            if neg_i_id not in pos_items and neg_i_id not in selected_items:
                break
        return neg_i_id


class KGTestGenerator(Dataset):
    def __init__(self, graph):
        self.users_to_test = list(graph.test_user_dict.keys())

    def __len__(self):
        return len(self.users_to_test)

    def __getitem__(self, index):
        batch_data = {}

        u_id = self.users_to_test[index]
        batch_data["u_id"] = u_id

        return batch_data


def KG_Data_loader(args_config, graph):
    '''
    load data for two samplers
    :param args_config:
    :param graph:
    :return: train_loader, test_loader
    '''
    train_generator = KGTrainGenerator(graph=graph)
    train_loader = DataLoader(
        train_generator,
        batch_size=args_config.batch_size,
        shuffle=True,
        num_workers=args_config.num_threads,
    )

    test_generator = KGTestGenerator(graph=graph)
    test_loader = DataLoader(
        test_generator,
        batch_size=args_config.test_batch_size,
        shuffle=False,
        num_workers=args_config.num_threads,
    )

    return train_loader, test_loader


def FM_Data_loader(mode, epoch=0):
    if mode == 'train':
        sample_file = bcfg.data_root + '/FM-sample-data/sample_fm_data_train-{}.pkl'.format(epoch)
    if mode == 'valid':
        sample_file = bcfg.data_root + '/FM-sample-data/sample_fm_data_valid.pkl'
    if mode == 'test':
        sample_file = bcfg.data_root + '/FM-sample-data/sample_fm_data_test.pkl'

    with open(sample_file, 'rb') as f:
        sample_data = pickle.load(f)

    return sample_data