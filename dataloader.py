import torch
from copy import deepcopy
import random
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Addr_Tx_Data:
    def __init__(self, addr_attrs, addr_classes, Tx_id2Input_Addr_ids, Tx_id2Output_Addr_ids, tx_attrs, tx_classes, tx_time_step):
        self.addr_attrs = self.__mean_normalization(addr_attrs)
        self.addr_classes = addr_classes
        self.Tx_id2Input_Addr_ids = Tx_id2Input_Addr_ids
        self.Tx_id2Output_Addr_ids = Tx_id2Output_Addr_ids
        self.tx_attrs = self.__mean_normalization(tx_attrs)
        self.tx_classes = tx_classes
        self.tx_time_step = tx_time_step
        self.__init_tx_id2addr_list()

    def __init_tx_id2addr_list(self):
        self.Tx_id2Addr_list = {}
        for tx_id in self.Tx_id2Input_Addr_ids:
            input_addr_ids, output_addr_ids = self.Tx_id2Input_Addr_ids[tx_id], self.Tx_id2Output_Addr_ids[tx_id]
            addr_ids = list(set(input_addr_ids + output_addr_ids))
            self.Tx_id2Addr_list[tx_id] = addr_ids
    
    def __mean_normalization(self, data):
        mean = np.mean(data, axis=0)
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - mean) / (max_val - min_val + 1e-6)
    
    def __z_score_normalization(self, data):
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


class Mask:
    # Mask for training, validation, and testing
    def __init__(self, train_addr, val_addr, test_addr, unknown_addr, train_tx, val_tx, test_tx, unknown_tx):
        self.train_addr, self.val_addr, self.test_addr = train_addr, val_addr, test_addr
        self.train_tx, self.val_tx, self.test_tx = train_tx, val_tx, test_tx
        self.unknown_addr, self.unknown_tx = unknown_addr, unknown_tx


def load_all_data():
    with open('dataset/Actors Dataset/data.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    Addr2Addr_id, Tx2Tx_id = loaded_dict['Addr2Addr_id'], loaded_dict['Tx2Tx_id']
    Tx_id2Output_Addr_ids, Tx_id2Input_Addr_ids = loaded_dict['Tx_id2Output_Addr_ids'], loaded_dict['Tx_id2Input_Addr_ids']

    addr_attrs = np.load('dataset/Actors Dataset/address_attrs.npy')
    addr_classes = np.load('dataset/Actors Dataset/address_classes.npy')

    tx_attrs = np.load('dataset/Actors Dataset/tx_attrs.npy')
    tx_classes = np.load('dataset/Actors Dataset/tx_classes.npy')
    tx_time_step = np.load('dataset/Actors Dataset/tx_time_step.npy')

    print(f"the node size is {addr_classes.shape[0]}, the tx size is {len(Tx_id2Input_Addr_ids)}")

    return Addr_Tx_Data(addr_attrs, addr_classes, Tx_id2Input_Addr_ids, Tx_id2Output_Addr_ids, tx_attrs, tx_classes, tx_time_step)


def sample_small_data(addr_attrs, addr_classes, Tx_id2Input_Addr_ids, Tx_id2Output_Addr_ids, tx_sample_size=1000):

    print(f"before selection, the node size is {addr_classes.shape[0]}, the tx size is {len(Tx_id2Input_Addr_ids)}")

    licit_ind, illicit_ind, _ = np.where(addr_classes == 0)[0], np.where(addr_classes == 1)[0], np.where(addr_classes == 2)[0]

    licit_ind_set, illicit_ind_set = set(licit_ind), set(illicit_ind)

    select_tx_ind_set = set()

    # choice txs that have both licit or illicit in input/output addresses
    for tx_id in Tx_id2Input_Addr_ids:
        input_addr_ids = Tx_id2Input_Addr_ids[tx_id]
        output_addr_ids = Tx_id2Output_Addr_ids[tx_id]
        if (len(input_addr_ids & licit_ind_set) > 0 or len(input_addr_ids & illicit_ind_set) > 0) and \
                (len(output_addr_ids & licit_ind_set) > 0 or len(output_addr_ids & illicit_ind_set) > 0):
            select_tx_ind_set.add(tx_id)

    random.seed(42)
    select_tx_ind_list = random.sample(list(select_tx_ind_set), tx_sample_size)
    select_node_ind_set = set()

    for select_tx_ind in select_tx_ind_list:
        select_node_ind_set.update(Tx_id2Input_Addr_ids[select_tx_ind])
        select_node_ind_set.update(Tx_id2Output_Addr_ids[select_tx_ind])

    select_node_ind_list = list(select_node_ind_set)

    print(f"after selection, the addr size is {len(select_node_ind_list)}, the tx size is {tx_sample_size}")
    
    select_node_ind2new_ind = {node_ind: i for i, node_ind in enumerate(select_node_ind_list)}
    select_tx_id2new_input_addr_ids = {tx_id: {select_node_ind2new_ind[node_ind] for node_ind in Tx_id2Input_Addr_ids[tx_id]} 
                                          for tx_id in select_tx_ind_list}
    select_tx_id2new_output_addr_ids = {tx_id: {select_node_ind2new_ind[node_ind] for node_ind in Tx_id2Output_Addr_ids[tx_id]} 
                                           for tx_id in select_tx_ind_list}
    select_node_attrs = addr_attrs[select_node_ind_list]
    select_node_classes = addr_classes[select_node_ind_list]
    return select_node_attrs, select_node_classes, select_tx_id2new_input_addr_ids, select_tx_id2new_output_addr_ids


def train_test_split_dataset(class_labels, train_ratio=0.4, val_ratio=0.2, test_ratio=0.4):
    train_idx, val_idx, test_idx = [], [], []
    licit_idx, illicit_idx, no_label = np.where(class_labels == 0)[0], np.where(class_labels == 1)[0], np.where(class_labels == 2)[0]
    print(f"licit ratio: {licit_idx.size / class_labels.size:.4f}, illicit ratio: {illicit_idx.size / class_labels.size:.4f}, no label ratio: {no_label.size / class_labels.size:.4f}")
    
    K_FOLD = 5
    for i in range(K_FOLD):
        licit_idx_train, licit_idx_val_test = train_test_split(licit_idx, train_size=0.4, random_state=i)
        illicit_idx_train, illicit_idx_val_test = train_test_split(illicit_idx, train_size=0.4, random_state=i)

        licit_idx_val, licit_idx_test = train_test_split(licit_idx_val_test, test_size=2.0/3, random_state=i)
        illicit_idx_val, illicit_idx_test = train_test_split(illicit_idx_val_test, test_size=2.0/3, random_state=i)
        train_idx.append(np.concatenate([licit_idx_train, illicit_idx_train]))
        val_idx.append(np.concatenate([licit_idx_val, illicit_idx_val]))
        test_idx.append(np.concatenate([licit_idx_test, illicit_idx_test]))
    return train_idx, val_idx, test_idx


def get_addr_classification_data(full_data: Addr_Tx_Data, train_ratio=0.2, transductive=False):
    addr_classes = full_data.addr_classes
    licit_addr_idx, illicit_addr_idx, unknown_addr_idx = np.where(addr_classes == 0)[0], np.where(addr_classes == 1)[0], np.where(addr_classes == 2)[0]
    print(f"licit ratio: {licit_addr_idx.size / addr_classes.size:.4f}, "
          f"illicit ratio: {illicit_addr_idx.size / addr_classes.size:.4f}, "
          f"no label ratio: {unknown_addr_idx.size / addr_classes.size:.4f}")

    licit_addr_idx_train, licit_addr_idx_val_test = train_test_split(licit_addr_idx, train_size=train_ratio, random_state=42)
    illicit_addr_idx_train, illicit_addr_idx_val_test = train_test_split(illicit_addr_idx, train_size=train_ratio, random_state=42)

    licit_addr_idx_val, licit_addr_idx_test = train_test_split(licit_addr_idx_val_test, test_size=2.0/3, random_state=42)
    illicit_addr_idx_val, illicit_addr_idx_test = train_test_split(illicit_addr_idx_val_test, test_size=2.0/3, random_state=42)
    
    train_addr_idx = np.concatenate([licit_addr_idx_train, illicit_addr_idx_train])
    val_addr_idx = np.concatenate([licit_addr_idx_val, illicit_addr_idx_val])
    test_addr_idx = np.concatenate([licit_addr_idx_test, illicit_addr_idx_test])

    if transductive:
        print('transductive setting addr classification')
        print(f'train addr size: {train_addr_idx.size}, val addr size: {val_addr_idx.size}, test addr size: {test_addr_idx.size}')
        return Mask(train_addr_idx, val_addr_idx, test_addr_idx, unknown_addr_idx, None, None, None, None)

    ## get tx_idx for training, validation and testing (Inductive)
    train_tx_idx, valid_tx_idx, test_tx_idx, unknown_tx_idx = [], [], [], []
    train_addr_set, valid_addr_set, test_addr_set = set(train_addr_idx) | set(unknown_addr_idx), set(val_addr_idx) | set(unknown_addr_idx), set(test_addr_idx) | set(unknown_addr_idx)
    for tx_id in full_data.Tx_id2Addr_list:
        addr_list = full_data.Tx_id2Addr_list[tx_id]
        if len(set(addr_list) & train_addr_set) == len(addr_list):
            train_tx_idx.append(tx_id)
        if len(set(addr_list) & valid_addr_set) == len(addr_list):
            valid_tx_idx.append(tx_id)
        if len(set(addr_list) & test_addr_set) == len(addr_list):
            test_tx_idx.append(tx_id)
        
        if full_data.tx_classes[tx_id] == 2:
            unknown_tx_idx.append(tx_id)
    
    print('inductive setting addr classification')

    print(f'train addr size: {train_addr_idx.size}, val addr size: {val_addr_idx.size}, test addr size: {test_addr_idx.size}')

    return Mask(train_addr_idx, val_addr_idx, test_addr_idx, unknown_addr_idx, 
                np.array(train_tx_idx).astype(int), np.array(valid_tx_idx).astype(int), 
                np.array(test_tx_idx).astype(int), np.array(unknown_tx_idx).astype(int))


def get_tx_classification_data(full_data: Addr_Tx_Data, train_ratio=0.2, time_split=True):
    tx_classes = full_data.tx_classes
    licit_tx_idx, illicit_tx_idx, unknown_tx_idx = \
            np.where(tx_classes == 0)[0], np.where(tx_classes == 1)[0], np.where(tx_classes == 2)[0]
    print(f"licit ratio: {licit_tx_idx.size / tx_classes.size:.4f}, "
          f"illicit ratio: {illicit_tx_idx.size / tx_classes.size:.4f}, "
          f"no label ratio: {unknown_tx_idx.size / tx_classes.size:.4f}")
    
    if not time_split:
        # split tx according to the class
        print('random split tx')
        licit_tx_idx_train, licit_tx_idx_val_test = train_test_split(licit_tx_idx, train_size=train_ratio, random_state=42)
        licit_tx_idx_val, licit_tx_idx_test = train_test_split(licit_tx_idx_val_test, test_size=2.0/3, random_state=42)

        illicit_tx_idx_train, illicit_tx_idx_val_test = train_test_split(illicit_tx_idx, train_size=train_ratio, random_state=42)
        illicit_tx_idx_val, illicit_tx_idx_test = train_test_split(illicit_tx_idx_val_test, test_size=2.0/3, random_state=42)

        train_tx_idx = np.concatenate([licit_tx_idx_train, illicit_tx_idx_train])
        val_tx_idx = np.concatenate([licit_tx_idx_val, illicit_tx_idx_val])
        test_tx_idx = np.concatenate([licit_tx_idx_test, illicit_tx_idx_test])

        print(f"train tx size: {train_tx_idx.size}, val tx size: {val_tx_idx.size}, test tx size: {test_tx_idx.size}")

        return Mask(None, None, None, None, train_tx_idx, val_tx_idx, test_tx_idx, unknown_tx_idx)

    else:
        # split tx according to the time
        print('split tx according to time')
        tx_time_step = full_data.tx_time_step
        label_tx_idx = np.where(tx_classes != 2)[0]
        label_tx_time_step = tx_time_step[tx_classes != 2]

        val_time, test_time = list(np.quantile(label_tx_time_step, [train_ratio, train_ratio + (1 - train_ratio) / 3]))

        train_tx_idx = label_tx_idx[label_tx_time_step < val_time]
        val_tx_idx = label_tx_idx[(val_time <= label_tx_time_step) & (label_tx_time_step < test_time)]
        test_tx_idx = label_tx_idx[test_time <= label_tx_time_step]

        print(f"train tx size: {train_tx_idx.size}, val tx size: {val_tx_idx.size}, test tx size: {test_tx_idx.size}")

        return Mask(None, None, None, None, train_tx_idx, val_tx_idx, test_tx_idx, unknown_tx_idx)


def get_link_prediction_data(full_data: Addr_Tx_Data, train_ratio=0.2):
    # split tx according to the time
    tx_time_step = full_data.tx_time_step
    tx_idx = np.arange(len(tx_time_step))
    val_time, test_time = list(np.quantile(tx_time_step, [train_ratio, train_ratio + (1 - train_ratio) / 3]))

    train_tx_idx = tx_idx[tx_time_step < val_time]
    val_tx_idx = tx_idx[(val_time <= tx_time_step) & (tx_time_step < test_time)]
    test_tx_idx = tx_idx[test_time <= tx_time_step]

    return Mask(None, None, None, train_tx_idx, val_tx_idx, test_tx_idx)


if __name__ == "__main__":
    load_raw_data()
