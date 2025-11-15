import dgl
import random
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from utils.utils import normalize, gen_dgl_graph
from dhg import Hypergraph


def get_hypergraph(neighbor_sampler):
    input_tx_e_set, output_tx_e_set = set(), set()
    input_e_list, output_e_list = [], []
    has_input_tx_ids, has_output_tx_ids = [], []
    for i in range(len(neighbor_sampler.tx2input_addr_list)):
        if len(neighbor_sampler.tx2input_addr_list[i]) > 0:
            input_addr_tuple = tuple(sorted(neighbor_sampler.tx2input_addr_list[i]))
            if input_addr_tuple not in input_tx_e_set:
                input_tx_e_set.add(input_addr_tuple)
                has_input_tx_ids.append(i)
                input_e_list.append(input_addr_tuple)

    hg_in = Hypergraph(num_v=len(neighbor_sampler.nodes_edge_ids), e_list=input_e_list)

    for i in range(len(neighbor_sampler.tx2output_addr_list)):
        if len(neighbor_sampler.tx2output_addr_list[i]) > 0:
            output_addr_tuple = tuple(sorted(neighbor_sampler.tx2output_addr_list[i]))
            if output_addr_tuple not in output_tx_e_set:
                output_tx_e_set.add(output_addr_tuple)
                has_output_tx_ids.append(i)
                output_e_list.append(output_addr_tuple)
    
    hg_out = Hypergraph(num_v=len(neighbor_sampler.nodes_edge_ids), e_list=output_e_list)

    assert len(hg_in.e[0]) == len(has_input_tx_ids)
    assert len(hg_out.e[0]) == len(has_output_tx_ids)

    return Transaction_Hypergraph(has_input_tx_ids=has_input_tx_ids,
                                  has_output_tx_ids=has_output_tx_ids,
                                  # tx_num=len(neighbor_sampler.tx2input_addr_list),
                                  tx_num = len(set(has_input_tx_ids) | set(has_output_tx_ids)),
                                  tx_time=neighbor_sampler.tx2time,
                                  addr_time=neighbor_sampler.node2time,
                                  addr_num=len(neighbor_sampler.nodes_edge_ids),
                                  hg_in=hg_in,
                                  hg_out=hg_out)


class Aug_Mask(object):
    def __init__(self, node_mask, edge_mask):
        # 1 for the preserved entity (node or hg edges), 0 for the dropped entity
        self.node_mask = node_mask
        self.edge_mask = edge_mask


class Augmentation:
    def __init__(self, task, aug1, aug2, rate=0.1):
        assert task in ['addr classification', 'tx classification']
        assert aug1 in ['identity', 'featmask', 'featdist', 'edgedrop', 'nodedrop', 'incidrop']
        assert aug2 in ['identity', 'featmask', 'featdist', 'edgedrop', 'nodedrop', 'incidrop']
        self.task = task
        self.aug1 = aug1
        self.aug2 = aug2
        self.rate = rate

    def generate(self, hg):
        hg_1, mask_1 = self.augment(hg, self.aug1)
        hg_2, mask_2 = self.augment(hg, self.aug2)
        return hg_1, mask_1, hg_2, mask_2

    def augment(self, hg, aug):
        if aug == 'identity':
            return self.identity(hg)
        elif aug == 'featmask':
            return self.featmask(hg)
        elif aug == 'featdist':
            return self.featdist(hg)
        elif aug == 'edgedrop':
            return self.edgedrop(hg)
        elif aug == 'nodedrop':
            return self.nodedrop(hg)
        elif aug == 'incidrop':
            return self.incidrop(hg)
        else:
            raise NotImplementedError

    def identity(self, hg):
        addr_num, edge_num = hg.addr_num, len(hg.subgraph_edge_ids)
        aug_mask = Aug_Mask(node_mask=torch.ones(addr_num), edge_mask=torch.ones(edge_num))
        return hg, aug_mask

    def featmask(self, hg):
        addr_num, edge_num = hg.addr_num, len(hg.subgraph_edge_ids)
        aug_mask = Aug_Mask(node_mask=torch.ones(addr_num), edge_mask=torch.ones(edge_num))
        return hg, aug_mask
    
    def featdist(self, hg):
        addr_num, edge_num = hg.addr_num, len(hg.subgraph_edge_ids)
        aug_mask = Aug_Mask(node_mask=torch.ones(addr_num), edge_mask=torch.ones(edge_num))
        return hg, aug_mask
    
    def edgedrop(self, hg):
        in_edge_num, out_edge_num = hg.hg_in.num_e, hg.hg_out.num_e
        addr_num, edge_num = hg.addr_num, len(hg.subgraph_edge_ids)
        hg_aug = deepcopy(hg)

        in_edge_sub = sorted(torch.randperm(in_edge_num)[: int((1 - self.rate) * in_edge_num)].tolist())
        out_edge_sub = sorted(torch.randperm(out_edge_num)[: int((1 - self.rate) * out_edge_num)].tolist())

        hg_aug_in_edge = [hg_aug.hg_in.e[0][i] for i in in_edge_sub]
        hg_aug_out_edge = [hg_aug.hg_out.e[0][i] for i in out_edge_sub]

        hg_aug.hg_in = Hypergraph(num_v=hg.addr_num, e_list=hg_aug_in_edge)
        hg_aug.hg_out = Hypergraph(num_v=hg.addr_num, e_list=hg_aug_out_edge)

        hg_aug.subgraph_in_edge_ids = [hg.subgraph_in_edge_ids[edge_id] for edge_id in in_edge_sub]
        hg_aug.subgraph_out_edge_ids = [hg.subgraph_out_edge_ids[edge_id] for edge_id in out_edge_sub]

        subgraph_edge_set = set(hg_aug.subgraph_in_edge_ids) | set(hg_aug.subgraph_out_edge_ids)

        hg_aug.subgraph_edge_ids = [edge_id for edge_id in hg.subgraph_edge_ids if edge_id in subgraph_edge_set]

        edge_mask_index = [i for i in range(len(hg.subgraph_edge_ids)) if hg.subgraph_edge_ids[i] in subgraph_edge_set]

        new_hg = Transaction_Hypergraph(has_input_tx_ids=hg_aug.subgraph_in_edge_ids,
                                        has_output_tx_ids=hg_aug.subgraph_out_edge_ids,
                                        tx_num=len(subgraph_edge_set),
                                        tx_time=hg.tx_time,
                                        addr_num=hg_aug.addr_num,
                                        addr_time=hg.addr_time,
                                        hg_in=hg_aug.hg_in,
                                        hg_out=hg_aug.hg_out,
                                        subgraph_node_ids=hg.subgraph_node_ids,
                                        subgraph_edge_ids=hg_aug.subgraph_edge_ids,
                                        subgraph_in_edge_ids=hg_aug.subgraph_in_edge_ids,
                                        subgraph_out_edge_ids=hg_aug.subgraph_out_edge_ids,
                                        label_idx=hg.label_idx if self.task=='addr classification' else [i for i in range(len(subgraph_edge_set))])

        edge_mask = torch.zeros(edge_num)
        edge_mask[edge_mask_index] = 1
        aug_mask = Aug_Mask(node_mask=torch.ones(addr_num), edge_mask=edge_mask)
        return new_hg, aug_mask

    def nodedrop(self, hg):
        addr_num, edge_num = hg.addr_num, len(hg.subgraph_edge_ids)
        addr_sub = torch.randperm(addr_num)[: int((1 - self.rate) * addr_num)]
        sub_hg = hg.sample_node(addr_sub.numpy())

        node_mask = torch.zeros(addr_num)
        node_mask[addr_sub] = 1

        sub_hg_edge_set = set(sub_hg.subgraph_edge_ids)
        edge_mask_index = [i for i in range(len(hg.subgraph_edge_ids)) if hg.subgraph_edge_ids[i] in sub_hg_edge_set]
        edge_mask = torch.zeros(edge_num)
        edge_mask[edge_mask_index] = 1
        sub_hg.subgraph_node_ids = [hg.subgraph_node_ids[node_id] for node_id in sub_hg.subgraph_node_ids]
        return sub_hg, Aug_Mask(node_mask, edge_mask)
    
    def incidrop(self, hg):
        addr_num, edge_num = hg.addr_num, len(hg.subgraph_edge_ids)
        hg_in_e, hg_out_e = deepcopy(hg.hg_in.e[0]), deepcopy(hg.hg_out.e[0])

        hg_in_e_set, hg_out_e_set = set(hg_in_e), set(hg_out_e)

        for i in range(len(hg_in_e)):
            in_e = list(hg_in_e[i])
            if len(in_e) > 1:
                flags = np.random.randint(0, 100, len(in_e)) > 100 * self.rate
                if not any(flags):
                    flags = np.random.randint(0, 100, len(in_e)) > 0
                new_in_e = tuple([in_e[j] for j in range(len(in_e)) if flags[j]])
                if new_in_e != hg_in_e[i] and new_in_e not in hg_in_e_set:
                    hg_in_e[i] = new_in_e
                    hg_in_e_set.add(new_in_e)
        
        for i in range(len(hg_out_e)):
            out_e = list(hg_out_e[i])
            if len(out_e) > 1:
                flags = np.random.randint(0, 100, len(out_e)) > 100 * self.rate
                if not any(flags):
                    flags = np.random.randint(0, 100, len(out_e)) > 0
                new_out_e = tuple([out_e[j] for j in range(len(out_e)) if flags[j]])
                if new_out_e != hg_out_e[i] and new_out_e not in hg_out_e_set:
                    hg_out_e[i] = new_out_e
                    hg_out_e_set.add(new_out_e)
        
        hg_in = Hypergraph(num_v=hg.addr_num, e_list=hg_in_e)
        hg_out = Hypergraph(num_v=hg.addr_num, e_list=hg_out_e)

        new_hg = Transaction_Hypergraph(has_input_tx_ids=hg.subgraph_in_edge_ids,
                                        has_output_tx_ids=hg.subgraph_out_edge_ids,
                                        tx_num=hg.tx_num,
                                        tx_time=hg.tx_time,
                                        addr_num=hg.addr_num,
                                        addr_time=hg.addr_time,
                                        hg_in=hg_in,
                                        hg_out=hg_out,
                                        subgraph_node_ids=hg.subgraph_node_ids,
                                        subgraph_edge_ids=hg.subgraph_edge_ids,
                                        subgraph_in_edge_ids=hg.subgraph_in_edge_ids,
                                        subgraph_out_edge_ids=hg.subgraph_out_edge_ids,
                                        label_idx=hg.label_idx)

        aug_mask = Aug_Mask(node_mask=torch.ones(addr_num), edge_mask=torch.ones(edge_num))
        return new_hg, aug_mask


class Transaction_Hypergraph:
    def __init__(self, has_input_tx_ids, has_output_tx_ids, tx_num, tx_time, addr_num, addr_time, hg_in, hg_out, 
                 subgraph_node_ids=None, subgraph_edge_ids=None, subgraph_in_edge_ids=None, subgraph_out_edge_ids=None, label_idx=None):
        self.addr_num = addr_num
        self.tx_num = tx_num
        self.tx_time = tx_time
        self.addr_time = addr_time
        self.has_input_tx_ids = has_input_tx_ids
        self.has_output_tx_ids = has_output_tx_ids
        self.hg_in = hg_in
        self.hg_out = hg_out
        self.subgraph_node_ids = subgraph_node_ids
        self.subgraph_edge_ids = subgraph_edge_ids
        self.subgraph_in_edge_ids = subgraph_in_edge_ids
        self.subgraph_out_edge_ids = subgraph_out_edge_ids
        self.label_idx = label_idx

        # cache
        self.has_input_tx_id_set = set(self.has_input_tx_ids)
        self.has_output_tx_id_set = set(self.has_output_tx_ids)

        self.has_v_to_input_e_id, self.in_tx_id2hg_in_eid = self.v_to_e_mapping(self.has_input_tx_ids, self.hg_in.e[0])
        self.has_v_to_output_e_id, self.out_tx_id2hg_out_eid = self.v_to_e_mapping(self.has_output_tx_ids, self.hg_out.e[0])
    
    def v_to_e_mapping(self, has_tx_ids, hg_edges):
        tx_id2hg_eid = {tx_id: i for i, tx_id in enumerate(has_tx_ids)}
        v_to_e_list = defaultdict(list)
        for tx_id, e in zip(has_tx_ids, hg_edges):
            for v in e:
                v_to_e_list[v].append(tx_id)
        return v_to_e_list, tx_id2hg_eid
    
    def sample_edge2hypergraph(self, ind):
        sorted_ind = sorted(ind)
        ind_set = set(ind)
        edge_ind2new_node_ind = {edge_id: i for i, edge_id in enumerate(sorted_ind)}
        
        new_edge_index_set = set()

        for edge_id in sorted_ind:
            if edge_id in self.has_input_tx_id_set:
                in_e = self.hg_in.e[0][self.in_tx_id2hg_in_eid[edge_id]]
                hyperedge = set()
                hyperedge.add(edge_ind2new_node_ind[edge_id])
                for v in in_e:
                    if v in self.has_v_to_input_e_id:
                        for tx_id in self.has_v_to_input_e_id[v]:
                            if tx_id != edge_id and tx_id in ind_set:
                                hyperedge.add(edge_ind2new_node_ind[tx_id])
                    if v in self.has_v_to_output_e_id:
                        for tx_id in self.has_v_to_output_e_id[v]:
                            if tx_id != edge_id and tx_id in ind_set:
                                hyperedge.add(edge_ind2new_node_ind[tx_id])
                if len(hyperedge) > 1:
                    new_edge_index_set.add(tuple(hyperedge))
            
            if edge_id in self.has_output_tx_id_set:
                out_e = self.hg_out.e[0][self.out_tx_id2hg_out_eid[edge_id]]
                hyperedge = set()
                hyperedge.add(edge_ind2new_node_ind[edge_id])
                for v in out_e:
                    if v in self.has_v_to_input_e_id:
                        for tx_id in self.has_v_to_input_e_id[v]:
                            if tx_id != edge_id and tx_id in ind_set:
                                hyperedge.add(edge_ind2new_node_ind[tx_id])
                    if v in self.has_v_to_output_e_id:
                        for tx_id in self.has_v_to_output_e_id[v]:
                            if tx_id != edge_id and tx_id in ind_set:
                                hyperedge.add(edge_ind2new_node_ind[tx_id])
                if len(hyperedge) > 1:
                    new_edge_index_set.add(tuple(hyperedge))

        new_edge_index_set = list(new_edge_index_set)
        hg = Hypergraph(num_v=len(ind), e_list=new_edge_index_set)                
        return hg, sorted_ind
    
    def sample_node2hypergraph(self, ind):
        sorted_ind = sorted(ind)
        ind_set = set(ind)
        node_ind2new_node_ind = {node_id: i for i, node_id in enumerate(sorted_ind)}

        new_node_index_set = set()

        for node_ind in sorted_ind:
            for tx_id in self.has_v_to_input_e_id[node_ind]:
                edge = self.hg_in.e[0][self.in_tx_id2hg_in_eid[tx_id]]
                hyperedge = set()
                hyperedge.add(node_ind2new_node_ind[node_ind])
                for v in edge:
                    if v != node_ind and v in ind_set:
                        hyperedge.add(node_ind2new_node_ind[v])
                if len(hyperedge) > 1:
                    new_node_index_set.add(tuple(hyperedge))
            for tx_id in self.has_v_to_output_e_id[node_ind]:
                edge = self.hg_out.e[0][self.out_tx_id2hg_out_eid[tx_id]]
                hyperedge = set()
                hyperedge.add(node_ind2new_node_ind[node_ind])
                for v in edge:
                    if v != node_ind and v in ind_set:
                        hyperedge.add(node_ind2new_node_ind[v])
                if len(hyperedge) > 1:
                    new_node_index_set.add(tuple(hyperedge))
        
        if len(new_node_index_set) == 0:
            for i in range(len(ind)):
                new_node_index_set.add((i))
        new_node_index_set = list(new_node_index_set)
        hg = Hypergraph(num_v=len(ind), e_list=new_node_index_set)
        return hg, sorted_ind

    def sample_node2graph(self, ind):
        sorted_ind = sorted(ind)
        ind_set = set(ind)
        node_ind2new_node_ind = {node_id: i for i, node_id in enumerate(sorted_ind)}

        new_node_index_set = set()

        for node_ind in sorted_ind:
            for tx_id in self.has_v_to_input_e_id[node_ind]:
                edge = self.hg_in.e[0][self.in_tx_id2hg_in_eid[tx_id]]
                for v in edge:
                    if v != node_ind and v in ind_set:
                        new_node_index_set.add((node_ind2new_node_ind[v], node_ind2new_node_ind[node_ind]))
                        new_node_index_set.add((node_ind2new_node_ind[node_ind], node_ind2new_node_ind[v]))
            for tx_id in self.has_v_to_output_e_id[node_ind]:
                edge = self.hg_out.e[0][self.out_tx_id2hg_out_eid[tx_id]]
                for v in edge:
                    if v != node_ind and v in ind_set:
                        new_node_index_set.add((node_ind2new_node_ind[v], node_ind2new_node_ind[node_ind]))
                        new_node_index_set.add((node_ind2new_node_ind[node_ind], node_ind2new_node_ind[v]))
        
        for i in range(len(ind)):
            new_node_index_set.add((i, i))
        edge_index1, edge_index2 = zip(*new_node_index_set)
                
        g = dgl.graph((edge_index1, edge_index2), num_nodes=len(ind))
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        adj = normalize(g.adj(), mode='sym')
        g = gen_dgl_graph(adj.indices()[0], adj.indices()[1], adj.values())
        return g, sorted_ind

    def sample_edge2graph(self, ind):
        sorted_ind = sorted(ind)
        ind_set = set(ind)
        edge_ind2new_node_ind = {edge_id: i for i, edge_id in enumerate(sorted_ind)}
        
        new_edge_index_set = set()

        for edge_id in sorted_ind:
            if edge_id in self.has_input_tx_id_set:
                in_e = self.hg_in.e[0][self.in_tx_id2hg_in_eid[edge_id]]
                for v in in_e:
                    if v in self.has_v_to_input_e_id:
                        for tx_id in self.has_v_to_input_e_id[v]:
                            if tx_id != edge_id and tx_id in ind_set:
                                new_edge_index_set.add((edge_ind2new_node_ind[tx_id], edge_ind2new_node_ind[edge_id]))
                                new_edge_index_set.add((edge_ind2new_node_ind[edge_id], edge_ind2new_node_ind[tx_id]))
                    if v in self.has_v_to_output_e_id:
                        for tx_id in self.has_v_to_output_e_id[v]:
                            if tx_id != edge_id and tx_id in ind_set:
                                new_edge_index_set.add((edge_ind2new_node_ind[tx_id], edge_ind2new_node_ind[edge_id]))
                                new_edge_index_set.add((edge_ind2new_node_ind[edge_id], edge_ind2new_node_ind[tx_id]))
            
            if edge_id in self.has_output_tx_id_set:
                out_e = self.hg_out.e[0][self.out_tx_id2hg_out_eid[edge_id]]
                for v in out_e:
                    if v in self.has_v_to_input_e_id:
                        for tx_id in self.has_v_to_input_e_id[v]:
                            if tx_id != edge_id and tx_id in ind_set:
                                new_edge_index_set.add((edge_ind2new_node_ind[tx_id], edge_ind2new_node_ind[edge_id]))
                                new_edge_index_set.add((edge_ind2new_node_ind[edge_id], edge_ind2new_node_ind[tx_id]))
                    if v in self.has_v_to_output_e_id:
                        for tx_id in self.has_v_to_output_e_id[v]:
                            if tx_id != edge_id and tx_id in ind_set:
                                new_edge_index_set.add((edge_ind2new_node_ind[tx_id], edge_ind2new_node_ind[edge_id]))
                                new_edge_index_set.add((edge_ind2new_node_ind[edge_id], edge_ind2new_node_ind[tx_id]))

        edge_index1, edge_index2 = zip(*new_edge_index_set)
                
        g = dgl.graph((edge_index1, edge_index2), num_nodes=len(ind))
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        adj = normalize(g.adj(), mode='sym')
        g = gen_dgl_graph(adj.indices()[0], adj.indices()[1], adj.values())
        return g, sorted_ind

    def sample_node(self, ind, hop=2):
        # sample node based on the input indices with hop distance
        ind_set = set(ind)
        filter_e_in, filter_e_out = set(), set()
        filter_e_in_list, filter_e_out_list = [], []
        filter_e_in_ids, filter_e_out_ids = [], []
        filter_n_in, filter_n_out = set(), set()

        union_in_v_set = ind_set & self.has_v_to_input_e_id.keys()
        union_out_v_set = ind_set & self.has_v_to_output_e_id.keys()

        # IN
        for v in union_in_v_set:
            for tx_id in self.has_v_to_input_e_id[v]:
                in_e = self.hg_in.e[0][self.in_tx_id2hg_in_eid[tx_id]]
                if in_e not in filter_e_in:
                    filter_e_in.add(in_e)
                    filter_e_in_list.append(in_e)
                    filter_e_in_ids.append(tx_id)
                    filter_n_in.update(set(in_e))
        
        for v in union_out_v_set:
            for tx_id in self.has_v_to_output_e_id[v]:
                out_e = self.hg_out.e[0][self.out_tx_id2hg_out_eid[tx_id]]
                if out_e not in filter_e_out:
                    filter_e_out.add(out_e)
                    filter_e_out_list.append(out_e)
                    filter_e_out_ids.append(tx_id)
                    filter_n_out.update(set(out_e))
        
        ind_set.update(filter_n_in)
        ind_set.update(filter_n_out)
        
        ind_list = sorted(list(ind_set))
        ind_node2idx = {v: i for i, v in enumerate(ind_list)}
        label_idx = [ind_node2idx[v] for v in ind]

        subgraph_edge_ids = sorted(list(set(filter_e_in_ids) | set(filter_e_out_ids)))

        filter_e_in_list_transformed = [tuple(ind_node2idx[v] for v in e) for e in filter_e_in_list]
        filter_e_out_list_transformed = [tuple(ind_node2idx[v] for v in e) for e in filter_e_out_list]

        subhg_in = Hypergraph(num_v=len(ind_list), e_list=filter_e_in_list_transformed)
        subhg_out = Hypergraph(num_v=len(ind_list), e_list=filter_e_out_list_transformed)

        return Transaction_Hypergraph(
            has_input_tx_ids=filter_e_in_ids,
            has_output_tx_ids=filter_e_out_ids,
            tx_num=len(set(filter_e_in_ids) | set(filter_e_out_ids)),
            tx_time=self.tx_time,
            addr_time=self.addr_time,
            addr_num=len(ind_list),
            hg_in=subhg_in, hg_out=subhg_out,
            subgraph_node_ids=ind_list,
            subgraph_edge_ids=subgraph_edge_ids,
            subgraph_in_edge_ids=filter_e_in_ids,
            subgraph_out_edge_ids=filter_e_out_ids,
            label_idx=label_idx
        )

    def sample_edge(self, ind):
        save_node_set = set()
        edge_set = set(ind)

        filter_e_in, filter_e_out = set(), set()
        filter_e_in_list, filter_e_out_list = [], []
        filter_e_in_ids, filter_e_out_ids = [], []
        filter_n_in, filter_n_out = set(), set()

        has_in_tx_ids_set2hg_in_index, has_out_tx_ids_set2hg_out_index = \
            {tx_id: i for i, tx_id in enumerate(self.has_input_tx_ids)}, {tx_id: i for i, tx_id in enumerate(self.has_output_tx_ids)}
        
        union_in_tx_ids = edge_set & self.has_input_tx_id_set
        union_out_tx_ids = edge_set & self.has_output_tx_id_set

        for in_tx_id in union_in_tx_ids:
            in_e = self.hg_in.e[0][has_in_tx_ids_set2hg_in_index[in_tx_id]]
            if in_e not in filter_e_in:
                filter_e_in.add(in_e)
                filter_e_in_list.append(in_e)
                filter_e_in_ids.append(in_tx_id)
                filter_n_in.update(set(in_e))

        for out_tx_id in union_out_tx_ids:
            out_e = self.hg_out.e[0][has_out_tx_ids_set2hg_out_index[out_tx_id]]
            if out_e not in filter_e_out:
                filter_e_out.add(out_e)
                filter_e_out_list.append(out_e)
                filter_e_out_ids.append(out_tx_id)
                filter_n_out.update(set(out_e))
        
        save_node_set.update(filter_n_in)
        save_node_set.update(filter_n_out)
        
        ind_list = sorted(list(save_node_set))
        ind_node2idx = {v: i for i, v in enumerate(ind_list)}
        ind_edge2idx = {e: i for i, e in enumerate(ind)}
        label_idx = [ind_edge2idx[e] for e in ind]

        filter_e_in_list_transformed = [tuple(ind_node2idx[v] for v in e) for e in filter_e_in_list]
        filter_e_out_list_transformed = [tuple(ind_node2idx[v] for v in e) for e in filter_e_out_list]

        subhg_in = Hypergraph(num_v=len(ind_list), e_list=filter_e_in_list_transformed)
        subhg_out = Hypergraph(num_v=len(ind_list), e_list=filter_e_out_list_transformed)

        return Transaction_Hypergraph(
            has_input_tx_ids=filter_e_in_ids,
            has_output_tx_ids=filter_e_out_ids,
            tx_num=len(set(filter_e_in_ids) | set(filter_e_out_ids)),
            tx_time=self.tx_time,
            addr_time=self.addr_time,
            addr_num=len(ind_list),
            hg_in=subhg_in, hg_out=subhg_out,
            subgraph_node_ids=ind_list,
            subgraph_edge_ids=ind,
            subgraph_in_edge_ids=filter_e_in_ids,
            subgraph_out_edge_ids=filter_e_out_ids,
            label_idx=label_idx
        )