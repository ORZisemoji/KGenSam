# construct source data to graph input data form
# ----------------------------------------------------------------------------------------------------
# ------------------------------------------- lastfm -------------------------------------------------
# ----------------------------------------------------------------------------------------------------
import json
import pickle
import pandas as pd
import numpy as np
# old_rootdir='/data/mengyuan/AUM-data'
rootdir='/home/mengyuan/KGenSam/data'
dataname='lastfm'
# tempdir= '/data/mengyuan/kgpolicy-data/last-fm'
############tool################################################
# def readtempdat(filename):
#     inter_mat = list()
#     path_kgpolicy = '/data/mengyuan/kgpolicy-data/last-fm/'
#     with open(path_kgpolicy+filename,'r') as f:
#         lines = f.readlines()
#         for l in lines:
#             tmps = l.strip()
#             inters = [int(i) for i in tmps.split(" ")]
#
#             u_id, pos_ids = inters[0], inters[1:]
#             pos_ids = list(set(pos_ids))
#
#             for i_id in pos_ids:
#                 inter_mat.append([u_id, i_id])
#     return inter_mat

def picklefile(data,filename):
    with open(filename,'wb') as f:
        pickle.dump(data,f)
    print('succeed to save {} !'.format(filename))

########################################################################################

fromdir='{}/{}/raw-data'.format(rootdir,dataname)
# FM_busi_list.pickle  FM_train_list.pickle  FM_valid_list.pickle
# FM_test_list.pickle  FM_user_list.pickle   item_dict.json
# review_dict_test.json  review_dict_valid.json  user_map.json
# item_map.json          review_dict_train.json  tag_map.json

todir='{}/{}/kgdata'.format(rootdir,dataname)
# user_list.dat
# item_list.dat
# relation_list.dat
# entity_list.dat
# kg_final.txt
# train.dat
# test.dat
# weights/
# pretrain_model/


with open('{}/user_list.pickle'.format(fromdir), 'rb') as f:
    user_list = pickle.load(f)
with open('{}/train_list.pickle'.format(fromdir), 'rb') as f:
    train_list = pickle.load(f)
with open('{}/valid_list.pickle'.format(fromdir), 'rb') as f:
    valid_list = pickle.load(f)
with open('{}/test_list.pickle'.format(fromdir), 'rb') as f:
    test_list = pickle.load(f)
with open('{}/item_dict.json'.format(fromdir), 'r') as f:
    item_dict = json.load(f)

# len(valid_list)/(len(train_list)+len(test_list)+len(valid_list))
# 0.2017263635533882
with open('{}/item_map.json'.format(fromdir), 'r') as f:
    item_map = json.load(f)

# kg_final.txt
# train.dat
# test.dat
##只需要构造以上三个文件
# train.dat

# temp_train=readtempdat('train.dat')  # [[u_id, pos_ids]......]

# temp_train=np.array(temp_train)
# 在这里构造时使用ear中的busi id与feature index
# 且存为numpy array
# id还是重复，需要再在preprocess中remap！！！！
all_list=train_list+valid_list+test_list
u_id,i_id=[],[]
for u,i in all_list:
    i_id.append(i)
    u_id.append(u)

print('range user id ; {} - {}'.format(min(u_id),max(u_id)))
print('range item id ; {} - {}'.format(min(i_id),max(i_id)))
u_id=list(set(u_id))
i_id=list(set(i_id))
# range user id ; 0 - 1800
# range item id ; 0 - 7430
print('len user id list; {} '.format(len(u_id)))
print('len item id list; {} '.format(len(i_id)))
# len user id list; 1801
# len item id list; 7123
picklefile(i_id,'{}/item_id_list.pickle'.format(fromdir))
picklefile(u_id,'{}/user_id_list.pickle'.format(fromdir))

i_id_list=[int(i) for i in list(item_dict.keys())]
print('len item id list in dic; {} '.format(len(i_id_list)))
# len item id list in dic; 7432
print('range item id in dic; {} - {}'.format(min(i_id_list),max(i_id_list)))
picklefile(i_id_list,'{}/item_id_list.pickle'.format(fromdir))

# feature str id to item id list
feature_dict=dict()
for i_id in item_dict.keys():
    values=item_dict[str(i_id)]
    for f_id in values['feature_index']:
        if str(f_id) not in feature_dict.keys():
            feature_dict[str(f_id)]=dict()
            feature_dict[str(f_id)]['item_index']=[i_id]
        else:
            feature_dict[str(f_id)]['item_index'].append(i_id)
picklefile(feature_dict,'{}/feature_dict.pickle'.format(fromdir))


kg_train=list()
# for u,i in train_list:
#     # kg_train.append([u,item_map_reverse[i]])  # [userid,busiid]
#     kg_train.append([u, i])
# for u,i in valid_list:
#     # kg_train.append([u,item_map_reverse[i]])  # [userid,busiid]
#     kg_train.append([u, i])
for u,i in all_list:
    # kg_train.append([u,item_map_reverse[i]])  # [userid,busiid]
    kg_train.append([u, i])
kg_train=np.array(kg_train)
# train.dat [u,i],u,i在kg中的id
picklefile(kg_train,'{}/train.dat'.format(todir))



# test.dat
kg_test=list()
for u,i in test_list:
    # kg_test.append([u,item_map_reverse[i]])  # [userid,busiid]
    kg_test.append([u, i])
kg_test=np.array(kg_test)
# test.dat
picklefile(kg_test,'{}/test.dat'.format(todir))


# kg_final.txt [i,r,a],i,r,a在kg中的id
# temp_kg_np = np.loadtxt('{}/kg_final.txt'.format(tempdir), dtype=np.int32)
# get triplets with canonical direction like <item, has-aspect, entity>
# 直接存为numpy array
# id不重复，不需要再在preprocess中remap属性id和边id！！！！

kg_final=[]
for k,v in item_dict.items():
    hid=int(k) # item的id
    rid=2 # 物品包含属性的关系
    for c in v['feature_index']:
        tid=c # cate的id
        kg_final.append([hid,rid,tid])
kg_final=np.array(kg_final)
picklefile(kg_final,'{}/kg_final.txt'.format(todir))



# ----------------------------------------------------------------------------------------------------
# ------------------------------------------- yelp ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------
import json
import pickle
import pandas as pd
import numpy as np
# old_rootdir='/data/mengyuan/AUM-data'
rootdir='/home/mengyuan/KGenSam/data'
dataname='yelp'
# tempdir= '/data/mengyuan/kgpolicy-data/last-fm'
############tool################################################
# def readtempdat(filename):
#     inter_mat = list()
#     path_kgpolicy = '/data/mengyuan/kgpolicy-data/last-fm/'
#     with open(path_kgpolicy+filename,'r') as f:
#         lines = f.readlines()
#         for l in lines:
#             tmps = l.strip()
#             inters = [int(i) for i in tmps.split(" ")]
#
#             u_id, pos_ids = inters[0], inters[1:]
#             pos_ids = list(set(pos_ids))
#
#             for i_id in pos_ids:
#                 inter_mat.append([u_id, i_id])
#     return inter_mat

def picklefile(data,filename):
    with open(filename,'wb') as f:
        pickle.dump(data,f)
    print('succeed to save {} !'.format(filename))

########################################################################################

fromdir='{}/{}/raw-data'.format(rootdir,dataname)
# FM_busi_list.pickle  FM_train_list.pickle  FM_valid_list.pickle
# FM_test_list.pickle  FM_user_list.pickle   item_dict.json
# review_dict_test.json  review_dict_valid.json  user_map.json
# item_map.json          review_dict_train.json  tag_map.json

todir='{}/{}/kgdata'.format(rootdir,dataname)
# user_list.dat
# item_list.dat
# relation_list.dat
# entity_list.dat
# kg_final.txt
# train.dat
# test.dat
# weights/
# pretrain_model/

with open('{}/review_dict_train.json'.format(fromdir, dir), 'rb') as f:
    _train_user_to_items = json.load(f)
with open('{}/review_dict_valid.json'.format(fromdir, dir), 'rb') as f:
    _valid_user_to_items = json.load(f)
with open('{}/review_dict_test.json'.format(fromdir, dir), 'rb') as f:
    _test_user_to_items = json.load(f)
_user_to_items=dict()
for u in _train_user_to_items.keys():
    _user_to_items[u]=_train_user_to_items[u]
    if u in _valid_user_to_items.keys():
        _user_to_items[u]=list(set(_user_to_items[u]+_valid_user_to_items[u]))
    if u in _test_user_to_items.keys():
        _user_to_items[u]=list(set(_user_to_items[u]+_test_user_to_items[u]))


with open('{}/user_list.pickle'.format(fromdir), 'rb') as f:
    user_list = pickle.load(f)
with open('{}/train_list.pickle'.format(fromdir), 'rb') as f:
    train_list = pickle.load(f)
with open('{}/valid_list.pickle'.format(fromdir), 'rb') as f:
    valid_list = pickle.load(f)
with open('{}/test_list.pickle'.format(fromdir), 'rb') as f:
    test_list = pickle.load(f)
with open('{}/item_dict-merge.json'.format(fromdir), 'r') as f:
    item_dict_merge = json.load(f)
with open('{}/item_dict-new.json'.format(fromdir), 'r') as f:
    item_dict_new = json.load(f)
item_dict=item_dict_merge
# with open('{}/item_dict-original_tag.json'.format(fromdir), 'r') as f:
#     item_dict_original_tag = json.load(f) # belong_to=(self.item, self.feature),
# with open('{}/item_dict-merged_tag.json'.format(fromdir), 'r') as f:
#     item_dict_merged_tag = json.load(f) # belong_to_large=(self.item, self.large_feature),
# with open('{}/2-layer taxonomy.json'.format(fromdir), 'r') as f:
#     layer_2_taxonomy = json.load(f) # link_to_feature=(self.large_feature, self.feature)




# len(valid_list)/(len(train_list)+len(test_list)+len(valid_list))
# 0.1993298290377216
with open('{}/item_map.json'.format(fromdir), 'r') as f:
    item_map = json.load(f)

# kg_final.txt
# train.dat
# test.dat
##只需要构造以上三个文件
# train.dat

# temp_train=readtempdat('train.dat')  # [[u_id, pos_ids]......]

# temp_train=np.array(temp_train)
# 在这里构造时使用ear中的busi id与feature index
# 且存为numpy array
# id还是重复，需要再在preprocess中remap！！！！
all_list=train_list+valid_list+test_list
u_id,i_id=[],[]
for u,i,r in all_list:
    i_id.append(i)
    u_id.append(u)

print('range user id ; {} - {}'.format(min(u_id),max(u_id)))
print('range item id ; {} - {}'.format(min(i_id),max(i_id)))
u_id=list(set(u_id))
i_id=list(set(i_id))
# range user id ; 0 - 27674
# range item id ; 0 - 70310
print('len user id list; {} '.format(len(u_id)))
print('len item id list; {} '.format(len(i_id)))
# len user id list; 27675
# len item id list; 66775
picklefile(i_id,'{}/item_id_list.pickle'.format(fromdir))
picklefile(u_id,'{}/user_id_list.pickle'.format(fromdir))

i_id_list=[int(i) for i in list(item_dict.keys())]
print('len item id list in dic; {} '.format(len(i_id_list)))
# len item id list in dic; 70311
picklefile(i_id_list,'{}/item_id_list.pickle'.format(fromdir))

# feature str id to item id list
feature_dict=dict()
for i_id in item_dict.keys():
    values=item_dict[str(i_id)]
    for f_id in values['feature_index']:
        if str(f_id) not in feature_dict.keys():
            feature_dict[str(f_id)]=dict()
            feature_dict[str(f_id)]['item_index']=[i_id]
        else:
            feature_dict[str(f_id)]['item_index'].append(i_id)
picklefile(feature_dict,'{}/feature_dict.pickle'.format(fromdir))


kg_train=list()
# for u,i in train_list:
#     # kg_train.append([u,item_map_reverse[i]])  # [userid,busiid]
#     kg_train.append([u, i])
# for u,i in valid_list:
#     # kg_train.append([u,item_map_reverse[i]])  # [userid,busiid]
#     kg_train.append([u, i])
for u,i,r in all_list:
    # kg_train.append([u,item_map_reverse[i]])  # [userid,busiid]
    kg_train.append([u, i])
kg_train=np.array(kg_train)
# train.dat [u,i],u,i在kg中的id
picklefile(kg_train,'{}/train.dat'.format(todir))



# test.dat
kg_test=list()
for u,i,r in test_list:
    # kg_test.append([u,item_map_reverse[i]])  # [userid,busiid]
    kg_test.append([u, i])
kg_test=np.array(kg_test)
# test.dat
picklefile(kg_test,'{}/test.dat'.format(todir))


# kg_final.txt [i,r,a],i,r,a在kg中的id
# temp_kg_np = np.loadtxt('{}/kg_final.txt'.format(tempdir), dtype=np.int32)
# get triplets with canonical direction like <item, has-aspect, entity>
# 直接存为numpy array
# id不重复，不需要再在preprocess中remap属性id和边id！！！！

kg_final=[]
for k,v in item_dict.items():
    hid=int(k) # item的id
    rid=2 # 物品包含属性的关系
    for c in v['feature_index']:
        tid=c # cate的id
        kg_final.append([hid,rid,tid])
kg_final=np.array(kg_final)
picklefile(kg_final,'{}/kg_final.txt'.format(todir))
