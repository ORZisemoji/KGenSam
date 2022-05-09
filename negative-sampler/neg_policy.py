import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as geometric
import networkx as nx

from tqdm import tqdm


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    Input: embedding matrix for knowledge graph entity and adjacency matrix
    Output: gcn embedding for kg entity
    """

    def __init__(self, in_channel, out_channel):
        super(GraphConv, self).__init__()

        self.conv1 = geometric.nn.SAGEConv(in_channel[0], out_channel[0])
        self.conv2 = geometric.nn.SAGEConv(in_channel[1], out_channel[1])

    def forward(self, x, edge_indices):
        x = self.conv1(x, edge_indices)
        x = F.leaky_relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_indices)
        x = F.dropout(x)
        x = F.normalize(x)

        return x


class KGPolicy(nn.Module):
    """
    Dynamical negative item negative-sampler based on Knowledge graph
    Input: user, postive item, knowledge graph embedding
    Ouput: qualified negative item
    """

    def __init__(self, rec, params, data_params):
        super(KGPolicy, self).__init__()
        self.params = params
        self.data_params = data_params
        self.rec = rec

        in_channel = eval(params.in_channel)
        out_channel = eval(params.out_channel)
        self.gcn = GraphConv(in_channel, out_channel)
        
        self.n_users=data_params['n_users']
        self.n_items=data_params['n_items']
        self.n_entities = data_params["n_nodes"]
        self.item_range = data_params["item_range"]
        self.input_channel = in_channel
        self.entity_embedding = self._initialize_weight(
            self.n_entities, self.input_channel
        )

    def _initialize_weight(self, n_entities, input_channel):
        """entities includes items and other entities in knowledge graph"""
        # if self.params.pretrain_s:
        #     kg_embedding = self.params["kg_embedding"]
        #     entity_embedding = nn.Parameter(kg_embedding)
        # else:
        #     entity_embedding = nn.Parameter(
        #         torch.FloatTensor(n_entities, input_channel[0])
        #     )
        #     nn.init.xavier_uniform_(entity_embedding)
        # 不在这里加载预训练模型
        entity_embedding = nn.Parameter(
            torch.FloatTensor(n_entities, input_channel[0])
        )
        nn.init.xavier_uniform_(entity_embedding)
        
        if self.params.freeze_s:
            entity_embedding.requires_grad = False

        return entity_embedding

    # def select_neg(self, user,pos, adj_matrix, edge_matrix):
    #     user = torch.tensor([user]).cuda(non_blocking=True)
    #     pos = torch.tensor([pos]).cuda(non_blocking=True)
    #     k = self.params.k_step
    #     assert k > 0
    #
    #     for _ in range(k):
    #         """sample candidate negative items based on knowledge graph"""
    #         one_hop, one_hop_logits = self.kg_step(pos, user, adj_matrix, step=1)
    #
    #         candidate_neg, two_hop_logits = self.kg_step(
    #             one_hop, user, adj_matrix, step=2
    #         )
    #         candidate_neg = self.filter_entity(candidate_neg, self.item_range)
    #         good_neg, good_logits = self.prune_step(
    #             self.rec, candidate_neg, user, two_hop_logits
    #         )
    #
    #         pos = good_neg
    #
    #     neg=good_neg
    #     print('select_neg:{}'.format(neg))
    #     return neg

    def forward(self, data_batch, adj_matrix, edge_matrix):
        users = data_batch["u_id"]
        pos = data_batch["pos_i_id"]

        self.edges = self.build_edge(edge_matrix)

        neg_list = torch.tensor([], dtype=torch.long, device=adj_matrix.device)
        prob_list = torch.tensor([], device=adj_matrix.device)

        k = self.params.k_step
        assert k > 0

        for _ in range(k):
            """sample candidate negative items based on knowledge graph"""
            one_hop, one_hop_logits = self.kg_step(pos, users, adj_matrix, step=1)

            candidate_neg, two_hop_logits = self.kg_step(
                one_hop, users, adj_matrix, step=2
            )
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.rec, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits + one_hop_logits

            # print('neg_list.shape:{};good_neg.shape:{}'.format(neg_list.shape,good_neg.shape))
            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])

            pos = good_neg

        return neg_list, prob_list

    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.params.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
            .unsqueeze(1)
            .repeat(1, sample_edge)
            .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    def kg_step(self, pos, user, adj_matrix, step):
        x = self.entity_embedding
        edges = self.edges

        """knowledge graph embedding using gcn"""
        gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)

        one_hop = adj_matrix[pos]
        i_e = gcn_embedding[one_hop]

        p_entity = F.leaky_relu(pos_e * i_e)
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.params.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits


    @staticmethod
    def prune_step(rec, negs, users, logits):
        # print('prune_step')
        with torch.no_grad():
            ranking = rec.rank(users, negs)

        # print('ranking.shape:{}'.format(ranking.shape))
        # print('ranking:{}'.format(ranking))
        """get most qualified negative item based on user-neg similarity"""
        indices = torch.argmax(ranking.cuda(), dim=1)

        batch_size = negs.size(0)
        row_id = torch.arange(batch_size, device=negs.device).unsqueeze(1)
        indices = indices.unsqueeze(1)

        good_neg = negs[row_id, indices].squeeze()
        goog_logits = logits[row_id, indices].squeeze()

        return good_neg, goog_logits

    @staticmethod
    def filter_entity(neg, item_range):
        random_neg = torch.randint(
            int(item_range[0]), int(item_range[1] + 1), neg.size(), device=neg.device
        )
        neg[neg > item_range[1]] = random_neg[neg > item_range[1]]
        neg[neg < item_range[0]] = random_neg[neg < item_range[0]]

        return neg
