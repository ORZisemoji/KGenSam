# data base

import sys

sys.path.append('/home/mengyuan/AUM-V4/FM')
from factorization_machine import FM
sys.path.append('/home/mengyuan/AUM-V4/user-simulator')
import json
import pickle
import time
import torch
import argparse

class _Base_Config():
    def __init__(self):
        self.dir_root='/home/mengyuan/AUM-V4'

        self.data_name = 'yelp'

        self.data_feature_two_layer=False
        if self.data_name=='lastfm':
            # false:only small features ; true: 2 layers feature
            self.data_feature_two_layer=False

        self.filename =''
        self.negSampler =1
        self.alSampler=1

        self.data_root=self.dir_root+'/data/'+self.data_name
        """ for RAP"""
        self.log_root=self.dir_root+'/run-log/'+self.data_name
        # self.log_root='/home/mengyuan/RAP'+'/run-log/'+self.data_name

        self.init_basic()

    def init_basic(self):

        dir = 'raw-data'

        # with open('{}/data/{}/review_dict_train.json'.format(self.data_root, dir), 'rb') as f:
        #     self._train_user_to_items = json.load(f)
        with open('{}/{}/review_dict_train.json'.format(self.data_root, dir), 'rb') as f:
            self._train_user_to_items = json.load(f)
        with open('{}/{}/review_dict_valid.json'.format(self.data_root, dir), 'rb') as f:
            self._valid_user_to_items = json.load(f)
        with open('{}/{}/review_dict_test.json'.format(self.data_root, dir), 'rb') as f:
            self._test_user_to_items = json.load(f)
        for u_id in self._train_user_to_items.keys():
            if u_id not in self._valid_user_to_items.keys():
                self._valid_user_to_items[u_id]=[]
            if u_id not in self._test_user_to_items.keys():
                self._test_user_to_items[u_id]=[]


        # self._user_to_items=dict()
        # for u in self._train_user_to_items.keys():
        #     self._user_to_items[u]=self._train_user_to_items[u]
        #     if u in self._valid_user_to_items.keys():
        #         self._user_to_items[u]=list(set(self._user_to_items[u]+self._valid_user_to_items[u]))
        #     if u in self._test_user_to_items.keys():
        #         self._user_to_items[u]=list(set(self._user_to_items[u]+self._test_user_to_items[u]))
        with open('{}/{}/user_item.json'.format(self.data_root, dir), 'rb') as f:
            self._user_to_items = json.load(f)

        with open('{}/{}/item_id_list.pickle'.format(self.data_root, dir), 'rb') as f:
            self.item_list = pickle.load(f)
            self.n_items=len(self.item_list)
        print('n_items is: {}'.format(self.n_items))
        with open('{}/{}/user_id_list.pickle'.format(self.data_root, dir), 'rb') as f:
            self.user_list = pickle.load(f)
            self.n_users=len(self.user_list)
        print('n_users is: {}'.format(self.n_users))

        with open('{}/{}/train_list.pickle'.format(self.data_root, dir), 'rb') as f:
            self.train_list = pickle.load(f)
        with open('{}/{}/valid_list.pickle'.format(self.data_root, dir), 'rb') as f:
            self.valid_list = pickle.load(f)
        with open('{}/{}/test_list.pickle'.format(self.data_root, dir), 'rb') as f:
            self.test_list = pickle.load(f)
        def _remove_r_4yelp(data_list):
            ui_list=[]
            for u,i,r in data_list:
                ui_list.append((u,i))
            return ui_list
        if self.data_name=='yelp':
            self.train_list = _remove_r_4yelp(self.train_list)
            self.valid_list = _remove_r_4yelp(self.valid_list)
            self.test_list = _remove_r_4yelp(self.test_list)

        # # _______ String to Int _______
        # with open('{}/data/{}/item_map.json'.format(self.data_root, dir), 'rb') as f:
        #     self.item_map = json.load(f)
        # with open('{}/data/{}/user_map.json'.format(self.data_root, dir), 'rb') as f:
        #     self.user_map = json.load(f)
        # with open('{}/data/{}/tag_map.json'.format(self.data_root, dir), 'rb') as f:
        #     self.tag_map = json.load(f)
        #     self.n_features=len(self.tag_map)
        #
        # self.tag_map_inverted = dict()
        # for k, v in self.tag_map.items():
        #     self.tag_map_inverted[v] = k  #int(feature id):str(cate id)

        # _______ item info _______
        with open('{}/{}/item_dict.pickle'.format(self.data_root, dir), 'rb') as f:
            self.item_dict = pickle.load(f)

        # _______ attri info _______
        with open('{}/{}/feature_dict.pickle'.format(self.data_root, dir), 'rb') as f:
            self.feature_dict = pickle.load(f)

        self.attribute_list=[]
        for k, v in self.item_dict.items():
            if v['feature_index']:
                self.attribute_list.extend(v['feature_index'])
                self.attribute_list=list(set(self.attribute_list))
        self.n_attributes = len(self.attribute_list)
        print('n_attributes is: {}'.format(self.n_attributes))

        if self.data_name=='yelp':
            with open('{}/{}/big_feature_dict.pickle'.format(self.data_root, dir), 'rb') as f:
                self.big_feature_dict = pickle.load(f)
            self.big_attribute_list=[]
            for k, v in self.item_dict.items():
                if v['big_feature_index']:
                    self.big_attribute_list.extend(v['big_feature_index'])
                    self.big_attribute_list=list(set(self.big_attribute_list))
            self.n_big_attributes = len(self.big_attribute_list)
            print('n_big_attributes is: {}'.format(self.n_big_attributes))

        self.PAD_IDX1 = self.n_users + self.n_items
        self.PAD_IDX2 = self.n_attributes

        ####################################################
        print("-" * 50)
        print('DATA NAME - {}'.format(self.data_name))
        print("-" * 50)
        print('train_list length is: {}'.format(len(self.train_list)))
        print('valid_list length is: {}'.format(len(self.valid_list)))
        print('test_list length is: {}'.format(len(self.test_list)))

        # #temp
        # kgdir= '/data/mengyuan/AUM-data/lastfm/kgdata'
        # with open('{}/busi_map_kg.dict'.format(kgdir), 'rb') as f:
        #     self.busi_map_kg = pickle.load(f)
        # self.kg_map_busi = dict()
        # for k, v in self.busi_map_kg.items():
        #     self.kg_map_busi[v] = k

    def get_FM_parser(self):
        parser = argparse.ArgumentParser(description="Run FM")
        parser.add_argument('-lr', default=0.02, type=float, dest='lr', help='lr')
        parser.add_argument('-flr', default=0.0001, type=float, dest='flr', help='flr')
        # means the learning rate of feature similarity learning
        parser.add_argument('-reg', default=0.001, type=float, dest='reg', help='reg')
        # regularization
        parser.add_argument('-decay', default=0.0, type=float, dest='decay', help='decay')
        # weight decay
        parser.add_argument('-bs', default=64, type=int, dest='bs', help='bs')
        # batch size
        parser.add_argument('-emb_size', default=64, type=int, dest='emb_size', help='emb_size')
        # hidden size/
        parser.add_argument('-ip', default=0.01, type=float, dest='ip', help='ip')
        # init parameter for hidden
        parser.add_argument('-dr', default=0.5, type=float, dest='dr', help='dr')
        # dropout ratio
        parser.add_argument('-optim', default='Ada', type=str, dest='optim', help='optim')
        # optimizer
        parser.add_argument('-observe', default=25, type=int, dest='observe', help='observe')
        # the frequency of doing evaluation
        parser.add_argument('-pretrain_epoch', default=0, type=int, dest='pretrain_epoch', help='pretrain_epoch')
        # does it need to load pretrain model
        parser.add_argument('-max_epoch', default=250, type=int, dest='max_epoch', help='max_epoch')
        # does it need to load pretrain model
        parser.add_argument('-updatefeatureemb', default=1, type=int, dest='updatefeatureemb', help='updatefeatureemb')
        # 0:不更新属性特征；1：更新
        parser.add_argument('-updateuseremb', default=1, type=int, dest='updateuseremb', help='updateuseremb')
        # 0:不更新用户特征；1：更新
        # parser.add_argument('-command', default=8,type=int, dest='command', help='command')
        # # command = 6: normal FM
        # # command = 8: with our second type of negative sample
        parser.add_argument('-seed', type=int, default=2021, dest='seed', help='seed')
        # random seed
        return parser.parse_args()


    def get_Negative_Sampler_parser(self):
        parser = argparse.ArgumentParser(description="Run KGbased-Sampler(KGPolicy).")
       # ------------- experimental settings specific for recommender ---------------------
        parser.add_argument(
            "-slr", type=float, default=0.001, dest='slr', help="Learning rate for negative-sampler."
        )
        parser.add_argument(
            "-rlr", type=float, default=0.001, dest='rlr', help="Learning rate recommender."
        )

        # ------------- experimental settings specific for negative-sampler -----------------------
        parser.add_argument(
            "-edge_threshold",
            type=int,
            default=64, dest='edge_threshold',
            help="edge threshold to filter knowledge graph",
        )
        parser.add_argument(
            "-num_sample", default=64, type=int, dest='num_sample', help="number fo samples from gcn"
        )
        parser.add_argument(
            "-k_step", default=1, type=int, dest='k_step', help="k step from current positive items"
        )
        parser.add_argument(
            "-in_channel", default="[64, 32]", type=str, dest='in_channel', help="input channels for gcn"
        )
        parser.add_argument(
            "-out_channel", default="[32, 64]", type=str, dest='out_channel', help="output channels for gcn"
        )

        # ------------- experimental settings specific for recommender ----------------------
        parser.add_argument(
            "-batch_size", type=int, default=64, dest='batch_size', help="batch size for training."
        )
        parser.add_argument(
            "-test_batch_size", type=int, default=64, dest='test_batch_size', help="batch size for test"
        )
        parser.add_argument("-num_threads", type=int, default=1, dest='num_threads', help="number of threads.")
        parser.add_argument("-epoch", type=int, default=400, dest='epoch', help="Number of epoch.")
        parser.add_argument("-show_step", type=int, default=3, dest='show_step', help="test step.")
        parser.add_argument(
            "-adj_epoch", type=int, default=1, dest='adj_epoch', help="build adj matrix per _ epoch"
        )
        parser.add_argument(
            "-pretrain_fm", type=bool, default=True, dest='pretrain_fm', help="use pretrained FM model or not"
        )
        parser.add_argument(
            "-pretrain_ns_epoch", type=int, default=0, dest='pretrain_ns_epoch', help="for breakpoint, use pretrained NS model or not"
        )
        parser.add_argument(
            "-freeze_s",
            type=bool,
            default=False, dest='freeze_s',
            help="freeze parameters of recoendAtmmender or not",
        )
        parser.add_argument("-flag_step", type=int, default=64, dest='flag_step', help="early stop steps")
        parser.add_argument(
            "-gamma", type=float, default=0.99, dest='gamma', help="gamma for reward accumulation"
        )

        # ------------- experimental settings specific for testing -----------------------
        parser.add_argument(
            "-Ks", nargs="?", default="[20, 40, 60, 80, 100]", dest='Ks', help="evaluate K list"
        )
        parser.add_argument('-seed', type=int, default=2021, dest='seed', help='seed')
        # random seed
        return parser.parse_args()

    def get_Active_Sampler_parser(self):
        parser = argparse.ArgumentParser(description="Run KGbased-Sampler(ALPolicy).")
        if self.data_name=='yelp':
            # 1:only small features ; 2: 2 layers feature
            parser.add_argument("-feature_layer", type=int, default=1, dest='feature_layer', help="Number of feature_layer.")
        parser.add_argument("-epoch", type=int, default=20000, dest='epoch', help="Number of epoch.")
        parser.add_argument(
            "-pretrain_fm", type=int, default=1, dest='pretrain_fm', help="use pretrained FM model or not"
        )
        parser.add_argument(
            "-pretrain_as_epoch", type=int, default=0, dest='pretrain_as_epoch',
            help="for breakpoint, use pretrained AS model or not"
        )
        parser.add_argument("--show_step", type=int, default=1000, help="valid step.")
        parser.add_argument('-mt', default=15, type=int, dest='mt', help='MAX_TURN')  # 15
        parser.add_argument("--nhid", type=int, default=64)
        parser.add_argument("--pnhid", type=list, default=[8, 8])
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--pdropout", type=float, default=0.0)
        parser.add_argument("--sllr", type=float, default=3e-2)
        parser.add_argument("--rllr", type=float, default=1e-2)
        parser.add_argument("--entcoef", type=float, default=0)
        parser.add_argument("--frweight", type=float, default=1e-3)
        parser.add_argument(
            "-batch_size", type=int, default=64, dest='batch_size', help="batch size for training."
        )
        # parser.add_argument("--budget", type=int, default=15, help="budget per class")
        parser.add_argument("--ntest", type=int, default=1000)
        parser.add_argument("--nval", type=int, default=500)
        parser.add_argument("--metric", type=str, default="microf1")

        parser.add_argument("--shaping", type=str, default="234", help="reward shaping method, 0 for no shaping;"
                                                                       "1 for add future reward,i.e. R= r+R*gamma;"
                                                                       "2 for use finalreward;"
                                                                       "3 for subtract baseline(value of curent state)"
                                                                       "1234 means all the method is used,")
        parser.add_argument("--logfreq", type=int, default=10)
        parser.add_argument("--policynet", type=str, default='mlp')
        parser.add_argument("--multigraphindex", type=int, default=0)

        parser.add_argument("--use_entropy", type=int, default=1)
        parser.add_argument("--use_degree", type=int, default=0)
        parser.add_argument("--use_local_diversity", type=int, default=0)
        parser.add_argument("--use_select", type=int, default=0)

        return parser.parse_args()



start = time.time()
print('___Begin Base Config ___')

bcfg = _Base_Config()
print('Base Config takes: {}'.format(time.time() - start))

print('___Base Config Done!!___')
