# Knowledge Graph-enhanced Sampling for Conversational Recommendation System

## Introduction
Knowledge Graph-enhanced Sampling for Conversational Recommendation System(KGenSam) is a Knowledge-enhanced framework tailored to conversational recommendation. KGenSam integrates the dynamic graph of user interaction data with the external knowledge into one heterogeneous Knowledge Graph(KG) as the contextual information environment. Then, two samplers are designed to enhance knowledge by sampling fuzzy samples with high uncertainty for obtaining user preferences and reliable negative samples for updating recommender to achieve efficient acquisition of user preferences and model updating, and thus provide a powerful solution for CRS to deal with E&E problem. 

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{KGenSam,
  author    = {Mengyuan Zhao, Xiaowen Huang, Lixi Zhu, Jitao Sang, Jian Yu},
  title     = {Knowledge Graph-enhanced Sampling for Conversational Recommender System},
  booktitle = {{TKDE}},
  pages     = {},
  year      = {}
}
```
## Environment Requirement
* Python >= 3.6
* Numpy >= 1.12
* PyTorch >= 1.0

## Example to Run the Code
The training models are saved in folder run-log/<data-name>/<training-model-name>-model.
The training logs are recorded in folder run-log/<data-name>/<training-model-name>-log and file run-log/<training-step>.out .

**0. Preparation**
```
python base_config.py 
```
The parser function of parameter settings is set in configuration/base_config.py. 
```
python knowledge_graph.py 
```
The knowledge graph is prepared in KG/knowledge_graph.py. 

**1. PreTrain FM**
```
python 1_fm_train.py
```
The implementation codes of FM model are in the folder FM.

**2. PreTrain Active Sampler and Negative Sampler**
```
python 2_active_sampler_train.py
python 2_negative_sampler_train.py
```
The Sampler implementation codes are in the folder active-sampler and the folder negative-sampler respectively.

**3. Train conversational Agent**
```
python 3_run.py
```
The implementation codes of conversational Agent are in the conversational-policy/conversational_policy.py .


**4. Evaluate conversational Agent**
```
python 4_policy_evaluate.py
```
The evaluation codes of conversational Agent are in the conversational-policy/conversational_policy_evaluate.py .

## Dataset
We provide two processed datasets: Last-FM, and Yelp2018.
* We follow [SCPR](https://github.com/farrecall/SCPR) to preprocess Last-FM and Yelp datasets.
* You can find the full version of recommendation datasets via [Last-FM]( https://grouplens.org/datasets/hetrec-2011/), [Yelp](https://www.yelp.com/dataset/)
* Here we list the relation types in different datasets to let readers to get
better understanding of the dataset.

<table>
  <tr>
    <th colspan="2">Dateset</th>
    <th>LastFM</th>
    <th>Yelp</th>
  </tr>
  <tr>
    <td rowspan="4">User-Item<br>Interaction</td>
    <td>#Users</td>
    <td>1,801</td>
    <td>27,675</td>
  </tr>
  <tr>
    <td>#Items</td>
    <td>7,432</td>
    <td>70,311</td>
  </tr>
  <tr>
    <td>#Interactions</td>
    <td>76,693</td>
    <td>1,368,606</td>
  </tr>
  <tr>
    <td>#attributes</td>
    <td>8,438</td>
    <td>590</td>
  </tr>
  <tr>
    <td rowspan="3">Graph</td>
    <td>#Entities</td>
    <td>17,671</td>
    <td>98,576</td>
  </tr>
  <tr>
    <td>#Relations</td>
    <td>4</td>
    <td>3</td>
  </tr>
  <tr>
    <td>#Triplets</td>
    <td>228,217</td>
    <td>2,533,827</td>
  </tr>
  <tr>
    <th>Relations</th>
    <th>Description</th>
    <th colspan="2">Number of Relations</th>
  </tr>
  <tr>
    <td>Interact</td>
    <td>user---item</td>
    <td>76,696</td>
    <td>1,368,606</td>
  </tr>
  <tr>
    <td>Friend</td>
    <td>user---user</td>
    <td>23,958</td>
    <td>688,209</td>
  </tr>
  <tr>
    <td>Like</td>
    <td>user---attribute</td>
    <td>33,120</td>
    <td>*</td>
  </tr>
  <tr>
    <td>Belong_to</td>
    <td>item---attribute</td>
    <td>94,446</td>
    <td>477,012</td>
  </tr>
</table>

**1. Graph Generate Data**

* `user_item.json`
  * Interaction file.
  * A dictionary of key value pairs. The key and the values of a dictionary entry: [`userID` : `a list of itemID`].
  
* `tag_map.json`
  * Map file.
  * A dictionary of key value pairs. The key and the value of a dictionary entry: [`Real attributeID` : `attributeID`].
  
* `user_dict.json`
  * User file.
  *  A dictionary of key value pairs. The key is `userID` and the value of a dictionary entry is a new dict: (''friends'' : `a list of userID`) & [''like'' : `attributeID`]
  
* `item_dict.json`
  * Item file.
  * A dictionary of key value pairs. The key is `itemID` and the value of a dictionary entry is a new dict: [''attribute_index'' : `a list of attributeID`] 

**2. FM Sample Data**
###### For the process of generating FM train data, please refer to Appendix B.2 of the paper.
* `sample_fm_data.pkl`
  *  The pickle file consists of five lists, and the fixed index of each list forms a training tuple`(user_id, item_id, neg_item, cand_neg_item, prefer_attributes)`.
           
  
```
user_pickle = pickle_file[0]           user id
item_p_pickle = pickle_file[1]         item id that has interacted with user
i_neg1_pickle = pickle_file[2]         negative item id that has not interacted with user
i_neg2_pickle = pickle_file[3]         negative item id that has not interacted with the user in the candidate item set
preference_pickle = pickle_file[4]     the user’s preferred attributes in the current turn
```

**3. UI Interaction Data**

* `review_dict.json`
    *  Items that the user has interacted with
    *  Used for generating FM sample data
    *  Used for training and testing in RL

**4. KG Data**

* `kg_final.txt`
    *  KG file.
    *  Used for generating KG data

## Acknowledgement
Any scientific publications that use our datasets should cite the following paper as the reference:
```
@inproceedings{KGenSam,
  author    = {Mengyuan Zhao, Xiaowen Huang, Lixi Zhu, Jitao Sang, Jian Yu},
  title     = {Knowledge Graph-enhanced Sampling for Conversational Recommender System},
  booktitle = {{TKDE}},
  year      = {}
}
```

Nobody guarantees the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:
* The user must acknowledge the use of the data set in publications resulting from the use of the data set.
* The user may not redistribute the data without separate permission.
* The user may not try to deanonymise the data.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.

## Funding Source Acknowledgement

This work is supported by the National Key R&D Program of China (2018AAA0100604), the Fundamental Research Funds for the Central Universities (2021RC217), the Beijing Natural Science Foundation (JQ20023), the National Natural Science Foundation of China (61632002, 61832004, 62036012, 61720106006).
