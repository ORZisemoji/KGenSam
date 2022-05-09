import collections
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import time
import sys
sys.path.append('/home/mengyuan/KGenSam/configuration')
from base_config import bcfg



# interaction data to graph:user-item
class _CFData(object):
    def __init__(self,data_name='lastfm'):
        path=bcfg.data_root
        train_file = path + "/kgdata/train.dat"
        test_file = path + "/kgdata/test.dat"

        self.item_id2kgid_map=dict()
        for i_id in bcfg.item_list:
            self.item_id2kgid_map[i_id]=i_id+bcfg.n_users
        self.attribute_id2kgid_map=dict()
        for a_id in bcfg.attribute_list:
            self.attribute_id2kgid_map[a_id]=a_id+bcfg.n_users+bcfg.n_items

        # ----------get number of users and items & then load interaction data from train_file & test_file------------
        self.train_data = self._generate_interactions(train_file)
        self.test_data = self._generate_interactions(test_file)

        self.train_user_dict, self.test_user_dict = self._generate_user_dict()

        self.exist_users,self.exist_items = self._generate_user_item_list()

        self._statistic_interactions()

    # reading train & test interaction data.
    @staticmethod
    def _generate_interactions(file_name):
        with open(file_name,'rb') as f:
            inter_mat=pickle.load(f)
        return inter_mat

    # generating user interaction dictionary.
    def _generate_user_dict(self):
        def _generate_dict(inter_mat):
            user_dict = dict()
            for u_id, i_id in inter_mat:
                if u_id not in user_dict.keys():
                    user_dict[u_id] = list()
                user_dict[u_id].append(self.item_id2kgid_map[i_id])
            return user_dict

        train_user_dict = _generate_dict(self.train_data)
        test_user_dict = _generate_dict(self.test_data)
        return train_user_dict, test_user_dict

    def _generate_user_item_list(self):
        inter_mat=np.concatenate((self.train_data,self.test_data),axis=0)
        user_list,item_list=[],[]
        for u_id, i_id in inter_mat:
            i_id=self.item_id2kgid_map[i_id]
            if u_id not in user_list:
                user_list.append(u_id)
            if i_id not in item_list:
                item_list.append(i_id)
        return user_list,item_list

    def _statistic_interactions(self):
        def _id_range(id_list):
            min_id = min(id_list)
            max_id = max(id_list)
            return (min_id, max_id), len(id_list)

        self.user_range, self.n_exist_users = _id_range(self.exist_users)
        self.item_range, self.n_exist_items = _id_range(self.exist_items)
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

        print("-" * 50)
        print("- INIT KG: user-item -")
        print("-" * 50)
        print("-     user_range: (%d, %d)" % (self.user_range[0], self.user_range[1]))
        print("-     item_range: (%d, %d)" % (self.item_range[0], self.item_range[1]))
        print("-        n_train: %d" % self.n_train)
        print("-         n_test: %d" % self.n_test)
        print("-        n_users: %d" % self.n_exist_users)
        print("-        n_items: %d" % self.n_exist_items)
        print("-" * 50)
        # --------------------------------------------------
        # -     user_range: (0, 1800)
        # -     item_range: (1801, 9231)
        # -        n_train: 68381
        # -         n_test: 8312
        # -        n_users: 1801
        # -        n_items: 7123
        # --------------------------------------------------

# items' attributes data to graph:item-attribute
class _KGData(object):
    def __init__(self, data_name='lastfm'):

        path=bcfg.data_root
        if bcfg.data_feature_two_layer: # for yelp
            kg_file = path + "/kgdata/kg_final_2_layers.txt"
        else:
            kg_file = path + "/kgdata/kg_final.txt"

        # ----------get number of entities and relations & then load kg data from kg_file ------------.
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)
        self._statistic_kg_triples()

    # reading train & test interaction data.
    def _load_kg(self, file_name):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        # get triplets with canonical direction like <item, has-aspect, entity>
        def _generate_kg_np(filename):
            with open(filename,'rb') as f:
                kg_np = pickle.load(f)
                kg_np = np.unique(kg_np, axis=0)
            new_kg_np=kg_np.copy()
            new_kg_np[:,0]=kg_np[:,0]+bcfg.n_users
            new_kg_np[:,2]=kg_np[:,2]+bcfg.n_users+bcfg.n_items
            return new_kg_np

        can_kg_np=_generate_kg_np(file_name)
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_kg_np = can_kg_np.copy()
        inv_kg_np[:, 0] = can_kg_np[:, 2]
        inv_kg_np[:, 2] = can_kg_np[:, 0]
        inv_kg_np[:, 1] = max(can_kg_np[:, 1]) + 1 # 3   # is-aspect-of id与原关系边id不重复。--2020.9.20

        # get full version of knowledge graph
        kg_np = np.concatenate((can_kg_np, inv_kg_np), axis=0)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _statistic_kg_triples(self):
        def _id_range(kg_mat, idx):
            min_id = min(min(kg_mat[:, idx]), min(kg_mat[:, 2 - idx]))
            max_id = max(max(kg_mat[:, idx]), max(kg_mat[:, 2 - idx]))
            n_id = max_id - min_id + 1
            return (min_id, max_id), n_id

        self.entity_range, self.n_entities = _id_range(self.kg_data, idx=0)
        self.relation_range, self.n_relations = _id_range(self.kg_data, idx=1)
        self.n_kg_triples = len(self.kg_data)

        print("-" * 50)
        print("- INIT KG: item-attribute -")
        print("-" * 50)
        print(
            "-   entity_range: (%d, %d)" % (self.entity_range[0], self.entity_range[1])
        )
        print(
            "- relation_range: (%d, %d)"
            % (self.relation_range[0], self.relation_range[1])
        )
        print("-     n_entities: %d" % self.n_entities)
        print("-    n_relations: %d" % self.n_relations)
        print("-   n_kg_triples: %d" % self.n_kg_triples)
        print("-" * 50)
        # --------------------------------------------------
        # -   entity_range: (1801, 9265)
        # - relation_range: (2, 3)
        # -     n_entities: 7465
        # -    n_relations: 2
        # -   n_kg_triples: 60580
        # --------------------------------------------------

# final graph:user-item-attribute
class _CKGData(_CFData, _KGData):
    def __init__(self,data_name='lastfm'):
        _CFData.__init__(self,data_name=data_name)
        _KGData.__init__(self,data_name=data_name)
        # itemid与userid不重复。--2020.9.20
        self.busi_list = bcfg.item_list
        self.n_items=bcfg.n_items
        self.user_list = bcfg.user_list
        self.n_users=bcfg.n_users

        self.ckg_graph = self._combine_cf_kg()

    def _combine_cf_kg(self):
        kg_mat = self.kg_data
        cf_mat = self.train_data

        # combine cf data and kg data:
        # ... ids of user entities in range of [0, #users)
        # ... ids of item entities in range of [#users, #users + #items)
        # ... ids of other entities in range of [#users + #items, #users + #entities)
        # ... ids of relations in range of [0, 1, 2 ,3], including two 'interact' and 'interacted_by'.
        ckg_graph = nx.MultiDiGraph()
        print("-" * 50)
        print("- COMBINE KG: user-item-attribute -")
        print("-" * 50)
        print("Begin to load interaction triples ...")
        for u_id, i_id in tqdm(cf_mat, ascii=True):
            ckg_graph.add_edges_from([(u_id, i_id)], r_id=0)
            ckg_graph.add_edges_from([(i_id, u_id)], r_id=1)

        print("\nBegin to load knowledge graph triples ...")
        for h_id, r_id, t_id in tqdm(kg_mat, ascii=True):
            ckg_graph.add_edges_from([(h_id, t_id)], r_id=r_id)
        return ckg_graph



start = time.time()
global_kg = _CKGData(data_name=bcfg.data_name)
print('Generate KG takes: {}'.format(time.time() - start))

print('___Generate KG as global_kg Done!!___')