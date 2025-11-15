import random
import torch_scatter
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from tgnn.modules import TimeEncoder
from utils.utils import NeighborSampler
from collections import defaultdict


class TDHConv(nn.Module):
    def __init__(self, hidden_dim, time_feat_dim, device):
        super(TDHConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_feat_dim = time_feat_dim
        self.device = device
        self.act2 = nn.LeakyReLU(inplace=True)
        
        # aggregation
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_e = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Linear(hidden_dim, 1, bias=False)

        self.lora_aggr = LoRALayer(hidden_dim * 2, hidden_dim)

        # propagation
        self.lora_prop = LoRALayer(hidden_dim * 2, hidden_dim)

    def edge_aggregation(self, edge_emb, node_emb, hg_edges):
        edge_size = torch.tensor([len(hg_edge) for hg_edge in hg_edges]).to(self.device)
        extend_edge_emb = torch.repeat_interleave(edge_emb, edge_size, dim=0)
        node_index = torch.tensor([i for hg_edge in hg_edges for i in hg_edge]).to(self.device)
        select_node_emb = torch.index_select(node_emb, 0, node_index)

        edge_node_attention = self.alpha(self.act2(self.linear_e(extend_edge_emb) + self.linear_v(select_node_emb)))

        edge_indices = torch.tensor([i for i in range(len(hg_edges)) for _ in range(len(hg_edges[i]))]).to(self.device)

        edge_node_attention_softmax = torch_scatter.scatter_softmax(edge_node_attention.squeeze(1), edge_indices, dim=0)

        edge_emb_aggr = edge_emb + torch_scatter.scatter_add(select_node_emb * edge_node_attention_softmax.unsqueeze(-1), edge_indices, dim=0)

        return edge_emb_aggr

    def edge_propagate(self, edge_emb, edge_time, node_emb, node_time, hg_edges, time_encoder):
        edge_time_feat = time_encoder(torch.tensor(edge_time).float().unsqueeze(1).to(self.device))
        node_time_feat = time_encoder(torch.tensor(node_time).float().unsqueeze(1).to(self.device))

        edge_emb_and_time = torch.cat([edge_emb, edge_time_feat], dim=1)
        node_emb_and_time = torch.cat([node_emb, node_time_feat], dim=1)

        node2edge_indices = defaultdict(list)

        for edge_indice, edge in enumerate(hg_edges):
            for node in edge:
                node2edge_indices[node].append(edge_indice)
        
        update_nodes, nodes_interleaved, node2edges = [], [], []
        for node in node2edge_indices:
            update_nodes.append(node)
            node2edges.append(node2edge_indices[node])
            nodes_interleaved.extend([node for _ in range(len(node2edge_indices[node]))])
        
        node2edges_1D = [edge_indice for edge_indices in node2edges for edge_indice in edge_indices]

        edge_attn_logits = torch.sum(edge_emb_and_time[node2edges_1D] * node_emb_and_time[nodes_interleaved], dim=-1) / math.sqrt(self.hidden_dim + self.time_feat_dim)

        edge_node_indice = torch.tensor([i for i in range(len(node2edges)) for _ in range(len(node2edges[i]))]).to(self.device)
        
        edge_attn = torch_scatter.scatter_softmax(edge_attn_logits, edge_node_indice, dim=0)

        edge_propagated_emb = torch_scatter.scatter_add(edge_emb[node2edges_1D] * edge_attn.unsqueeze(-1), edge_node_indice, dim=0)

        return edge_propagated_emb, update_nodes
    
    def forward(self, node_emb, edge_emb, input_edge_indices, output_edge_indices, sub_hgraph, time_encoder):
        # edge aggregation
        input_edge_emb, output_edge_emb = edge_emb[input_edge_indices], edge_emb[output_edge_indices]

        input_edge_emb_aggr = self.edge_aggregation(input_edge_emb, node_emb, sub_hgraph.hg_in.e[0])
        output_edge_emb_aggr = -self.edge_aggregation(output_edge_emb, node_emb, sub_hgraph.hg_out.e[0])

        edge_emb_new = torch_scatter.scatter_sum(torch.cat([edge_emb, input_edge_emb_aggr, output_edge_emb_aggr], dim=0),
                                             torch.cat([torch.tensor([i for i in range(len(sub_hgraph.subgraph_edge_ids))]).to(self.device), input_edge_indices, output_edge_indices], dim=0), dim=0)

        # edge propagation
        input_node_emb_prop, update_in_nodes = self.edge_propagate(edge_emb=edge_emb_new[input_edge_indices],
                                                                   edge_time=sub_hgraph.tx_time[sub_hgraph.subgraph_in_edge_ids],
                                                                   node_emb=node_emb,
                                                                   node_time=np.array(sub_hgraph.addr_time)[sub_hgraph.subgraph_node_ids],
                                                                   hg_edges=sub_hgraph.hg_in.e[0],
                                                                   time_encoder=time_encoder)
        
        output_node_emb_prop, update_out_nodes = self.edge_propagate(edge_emb=edge_emb_new[output_edge_indices],
                                                                     edge_time=sub_hgraph.tx_time[sub_hgraph.subgraph_out_edge_ids],
                                                                     node_emb=node_emb,
                                                                     node_time=np.array(sub_hgraph.addr_time)[sub_hgraph.subgraph_node_ids],
                                                                     hg_edges=sub_hgraph.hg_out.e[0],
                                                                     time_encoder=time_encoder)

        node_emb[update_in_nodes] += input_node_emb_prop
        node_emb[update_out_nodes] -= output_node_emb_prop
        return node_emb, edge_emb


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / rank)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 1.0

    def forward(self, x):
        return (self.lora_B @ self.lora_A @ x.T).T * self.scaling


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert self.hidden_dim % self.num_heads == 0
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = nn.Dropout(p=dropout)

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, mask):
        node_size, _ = x.size()

        q = self.q_linear(x).view(node_size, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_linear(x).view(node_size, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_linear(x).view(node_size, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_probs, v)
        output = output.transpose(1, 2).contiguous().view(node_size, self.hidden_dim)
        output = self.dropout(self.out_proj(output)) + x

        return output


class BitGAT(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                 time_feat_dim: int, hidden_dim: int, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, atten_neg_slope: float = 0.2, device: str = 'cpu'):
        super(BitGAT, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.time_feat_dim = time_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        self.node_linear = nn.Linear(self.node_raw_features.size(1), hidden_dim)
        self.edge_linear = nn.Linear(self.edge_raw_features.size(1), hidden_dim)

        # self.atten_dropout = nn.Dropout(dropout)
        # self.atten_act = nn.LeakyReLU(atten_neg_slope)
        # self.atten_linear = nn.Linear(hidden_dim, 1, bias=False)
        # self.atten_dst = nn.Linear(hidden_dim, 1, bias=False)
        # self.self_attention = SelfAttention(hidden_dim, num_heads, dropout)
        self.act = nn.LeakyReLU()

        # time-encoder
        self.time_encoder = TimeEncoder(time_feat_dim)

        self.tier_convs = nn.ModuleList([TDHConv(self.hidden_dim, self.time_feat_dim, self.device) for _ in range(num_layers)])

    def forward(self, sub_hgraph, task):
        assert task in ['addr classification', 'tx classification', 'link prediction']
        node_emb, edge_emb, label_id = self.classification(sub_hgraph)
        if task == 'addr classification':
            return node_emb[label_id]
        elif task == 'tx classification':
            return edge_emb[label_id]

    def cl_forward(self, sub_hgraph, aug, rate, task):
        node_emb, edge_emb, label_id = self.classification(sub_hgraph, aug=aug, rate=rate)
        if task == 'addr classification':
            if aug == 'nodedrop':
                return node_emb[label_id], edge_emb
            else:
                return node_emb, edge_emb
        elif task == 'tx classification':
            if aug == 'nodedrop':
                return node_emb[label_id], edge_emb
            else:
                return node_emb, edge_emb[label_id]

    def node_self_attention(self, node_emb, hg_edges, max_num_neighbors=10):
        # obtain the neighbors of each nodes
        node_neighbors = defaultdict(set)

        for edge_tuple in hg_edges:
            for i in range(len(edge_tuple)):
                for j in range(len(edge_tuple)):
                    node_neighbors[edge_tuple[i]].add(edge_tuple[j])

        for node in node_neighbors:
            node_neighbors[node] = list(node_neighbors[node])

        mask = torch.eye(len(node_emb)).to(self.device)

        for node in node_neighbors:
            mask[node, node_neighbors[node]] = 1

        node_emb_attn = self.self_attention(node_emb, mask)
        return node_emb_attn

    def feat_drop(self, x, rate):
        total_elements = x.numel()
        num_elements_to_mask = int(total_elements * rate)
        mask_indices = np.random.choice(total_elements, num_elements_to_mask, replace=False)
        mask = torch.ones_like(x).flatten()
        mask[mask_indices] = 0
        mask = mask.view(x.shape)
        masked_matrix = x.clone() * mask
        return masked_matrix

    def feat_pert(self, x, rate, noise_std=0.1):
        total_elements = x.numel()
        num_elements_to_perturb = int(total_elements * rate)
        perturb_indices = np.random.choice(total_elements, num_elements_to_perturb, replace=False)
        flat_matrix = x.flatten()
        perturbed_matrix = flat_matrix.clone()
        noise = torch.randn(num_elements_to_perturb) * noise_std
        perturbed_matrix[perturb_indices] += noise
        perturbed_matrix = perturbed_matrix.view(x.shape)
        return perturbed_matrix

    def classification(self, sub_hgraph, aug=None, rate=0.1):
        edge_id2indices = {}
        for edge_id in sub_hgraph.subgraph_edge_ids:
            if edge_id not in edge_id2indices:
                edge_id2indices[edge_id] = len(edge_id2indices)

        sample_node_indices = torch.from_numpy(np.array(sub_hgraph.subgraph_node_ids)).to(self.device)
        sample_edge_indices = torch.from_numpy(np.array(sub_hgraph.subgraph_edge_ids)).to(self.device)
        input_edge_indices = torch.from_numpy(np.array([edge_id2indices[edge_id] for edge_id in sub_hgraph.subgraph_in_edge_ids])).to(self.device)
        output_edge_indices = torch.from_numpy(np.array([edge_id2indices[edge_id] for edge_id in sub_hgraph.subgraph_out_edge_ids])).to(self.device)
        label_id = torch.from_numpy(np.array(sub_hgraph.label_idx)).to(self.device)

        input_node_features = self.node_raw_features[sample_node_indices].clone()
        input_edge_features = self.edge_raw_features[sample_edge_indices].clone()
        
        if aug == 'featmask':
            input_node_features = self.feat_drop(input_node_features.to('cpu'), rate).to(self.device)
            input_edge_features = self.feat_drop(input_edge_features.to('cpu'), rate).to(self.device)
        elif aug == 'featdist':
            input_node_features = self.feat_pert(input_node_features.to('cpu'), rate).to(self.device)
            input_edge_features = self.feat_pert(input_edge_features.to('cpu'), rate).to(self.device)

        node_emb = self.act(self.node_linear(input_node_features))
        edge_emb = self.act(self.edge_linear(input_edge_features))

        for i in range(self.num_layers):
            node_emb, edge_emb = self.tier_convs[i](node_emb, edge_emb, input_edge_indices, output_edge_indices, sub_hgraph, self.time_encoder)
            node_emb, edge_emb = self.act(node_emb), self.act(edge_emb)

        return node_emb, edge_emb, label_id
