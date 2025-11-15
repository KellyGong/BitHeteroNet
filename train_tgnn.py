import os
import dgl
import numpy as np
from copy import deepcopy
import torch
from tgnn import *
import torch.nn as nn
from utils import *
from contextlib import contextmanager
import torch.nn.functional as F

from tqdm import tqdm
from gnn import get_gnn_model
from hypergraph import get_hypergraph, Augmentation


def pretrain_epoch(args, full_data, model, optimizer, augmentor, loss_func, epoch, idx_data_loader, hgraph, mode='train'):
    assert mode in ['train', 'val']
    total_loss = 0.0
    
    tbar = tqdm(idx_data_loader, ncols=180)

    with conditional_no_grad(mode in ['val']):
        for batch_idx, data_indices in enumerate(tbar):
            try:
                loss_node, loss_edge = cl_batch_inference(args, full_data, model, augmentor, loss_func, data_indices, hgraph)
                loss = loss_node + loss_edge
                tbar.set_description(f'Epoch: {epoch + 1}, {mode} for the {batch_idx + 1} batch, {mode} batch loss: {loss.item():.4f} '
                                    f'node loss :{loss_node.item():.4f}, edge loss: {loss_edge.item():.4f}')
                total_loss += loss.item()
         
                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            except Exception:
                continue

    total_loss /= (batch_idx + 1)
    return total_loss


def cl_batch_inference(args, full_data, model, augmentor, loss_func, data_indices, hgraph):
    data_indices = data_indices.numpy()
    if args.task == 'addr classification':
        sub_hgraph = hgraph.sample_node(data_indices)
    else:
        sub_hgraph = hgraph.sample_edge(data_indices)

    hg_1, mask_1, hg_2, mask_2 = augmentor.generate(sub_hgraph)

    batch_node_embeddings_1, batch_edge_embeddings_1 = model[0].cl_forward(hg_1, args.aug1, args.pretrain_rate, args.task)

    batch_node_embeddings_2, batch_edge_embeddings_2 = model[0].cl_forward(hg_2, args.aug2, args.pretrain_rate, args.task)

    proj_node_1 = model[1](batch_node_embeddings_1)
    proj_node_2 = model[1](batch_node_embeddings_2)

    proj_edge_1 = model[2](batch_edge_embeddings_1)
    proj_edge_2 = model[2](batch_edge_embeddings_2)

    loss_node = loss_func(proj_node_1, proj_node_2, mask_1.node_mask, mask_2.node_mask)

    loss_edge = loss_func(proj_edge_1, proj_edge_2, mask_1.edge_mask, mask_2.edge_mask)

    return loss_node, loss_edge


def pretrain_loss(proj_emb_1, proj_emb_2, mask_1, mask_2):
    assert int(mask_1.sum()) == proj_emb_1.size(0)
    assert int(mask_2.sum()) == proj_emb_2.size(0)
    similarity = proj_emb_1 @ proj_emb_2.t()
    
    mask_1_index = torch.nonzero(mask_1).squeeze(1)
    mask_2_index = torch.nonzero(mask_2).squeeze(1)

    mask = mask_1.long() & mask_2.long()

    mask_index_list = torch.nonzero(mask).squeeze(1).tolist()
    mask_1_index_list = mask_1_index.tolist()
    mask_2_index_list = mask_2_index.tolist()

    row_index, col_index = [], []
    mask_1_index2id = {mask_1_index_list[i]: i for i in range(len(mask_1_index_list))}
    mask_2_index2id = {mask_2_index_list[i]: i for i in range(len(mask_2_index_list))}
    for i in range(len(mask_index_list)):
        row_index.append(mask_1_index2id[mask_index_list[i]])
        col_index.append(mask_2_index2id[mask_index_list[i]])

    label_matrix = torch.zeros((proj_emb_1.size(0), proj_emb_2.size(0))).to(proj_emb_1.device)
    label_matrix[row_index, col_index] = 1
    
    loss = ((F.binary_cross_entropy(F.softmax(similarity, dim=0), label_matrix, reduction='none') * label_matrix).sum() / len(row_index) + \
            (F.binary_cross_entropy(F.softmax(similarity, dim=1), label_matrix, reduction='none') * label_matrix).sum() / len(row_index)) / 2

    return loss


def pretrain_tgnn(args, dynamic_backbone, full_data, train_idx_data_loader, train_hgraph, val_idx_data_loader, valid_hgraph):
    print(f'Pretraining... with {args.pretrain_epochs} epochs, aug1 {args.aug1}, aug2 {args.aug2}.')
    projector_node = Projector(args.hid_dim, args.dropout)
    projector_edge = Projector(args.hid_dim, args.dropout)
    pretrain_model = nn.Sequential(dynamic_backbone, projector_node, projector_edge)
    pretrain_model = convert_to_gpu(pretrain_model, device=args.device)
    pretrain_optimizer = create_optimizer(model=pretrain_model, optimizer_name=args.optimizer, learning_rate=args.pretrain_lr, weight_decay=args.pretrain_w_decay)
    
    best_val_loss = float('inf')
    best_model_state = None

    loss_func = pretrain_loss

    augmentor = Augmentation(args.task, args.aug1, args.aug2, args.pretrain_rate)

    for epoch in range(args.pretrain_epochs):
        pretrain_model.train()
        train_epoch_loss = pretrain_epoch(args, full_data, pretrain_model, pretrain_optimizer, augmentor, loss_func, epoch, train_idx_data_loader, train_hgraph, mode='train')

        pretrain_model.eval()
        val_epoch_loss = pretrain_epoch(args, full_data, pretrain_model, pretrain_optimizer, augmentor, loss_func, epoch, val_idx_data_loader, valid_hgraph, mode='val')

        print(f'Pretraining epoch: {epoch}, train epoch loss: {train_epoch_loss:.4f}, val epoch loss: {val_epoch_loss:.4f}')

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            early_stop = 0
            best_model_state = deepcopy(pretrain_model.state_dict())

        else:
            early_stop += 1

        if early_stop > args.patience:
            break
    
    pretrain_model.load_state_dict(best_model_state)
    return dynamic_backbone


def train_eval_tgnn(args, full_data, mask, fold_seed):
    set_random_seed(fold_seed)

    train_neighbor_sampler = get_neighbor_sampler(data=full_data, 
                                                  addr_mask=np.concatenate((mask.train_addr, mask.unknown_addr)) if mask.train_addr is not None else None, 
                                                  tx_mask=mask.train_tx,
                                                  seed=1)

    train_hgraph = get_hypergraph(train_neighbor_sampler)
    
    valid_neighbor_sampler = get_neighbor_sampler(data=full_data, addr_mask=np.concatenate((mask.val_addr, mask.unknown_addr)) if mask.val_addr is not None else None, 
                                                  tx_mask=mask.val_tx,
                                                  seed=1)

    valid_hgraph = get_hypergraph(valid_neighbor_sampler)

    test_neighbor_sampler = get_neighbor_sampler(data=full_data, addr_mask=np.concatenate((mask.test_addr, mask.unknown_addr)) if mask.test_addr is not None else None,
                                                 tx_mask=mask.test_tx,
                                                 seed=1)
    
    test_hgraph = get_hypergraph(test_neighbor_sampler)

    print(f'train graph addr num: {train_hgraph.addr_num}, train graph tx num: {train_hgraph.tx_num}')
    print(f'valid graph addr num: {valid_hgraph.addr_num}, valid graph tx num: {valid_hgraph.tx_num}')
    print(f'test graph addr num: {test_hgraph.addr_num}, test graph tx num: {test_hgraph.tx_num}')

    # create model
    if args.tgnn == 'BitGAT':
        dynamic_backbone = BitGAT(node_raw_features=full_data.addr_attrs, edge_raw_features=full_data.tx_attrs,
                                   time_feat_dim=args.time_feat_dim, hidden_dim=args.hid_dim, num_heads=args.num_heads,
                                   num_layers=args.num_layers, dropout=args.dropout, device=args.device)
    
    elif args.tgnn in ['gcn', 'gpr', 'gat', 'gin', 'appnp', 'bwgnn', 'HGNN', 'HGNN+', 'HNHN', 'UniGAT']:
        in_dim = full_data.addr_attrs.shape[1] if args.task == 'addr classification' else full_data.tx_attrs.shape[1]
        dynamic_backbone = get_gnn_model(model_str=args.tgnn,
                                         in_channels=in_dim,
                                         hidden_channels=args.hid_dim,
                                         out_channels=args.hid_dim,
                                         dropout=args.dropout,
                                         dropout_adj=args.dropout_adj,
                                         num_layers=args.num_layers)

    else:
        raise ValueError(f'Invalid model name: {args.tgnn}')

    if args.task == 'addr classification':
        addr_classifier = MLPClassifier(input_dim=args.hid_dim, dropout=args.dropout)
        model = nn.Sequential(dynamic_backbone, addr_classifier)
        train_idx_data_loader = get_idx_data_loader(mask.train_addr, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_idx_data_loader = get_idx_data_loader(mask.val_addr, batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_idx_data_loader = get_idx_data_loader(mask.test_addr, batch_size=args.batch_size, shuffle=False, drop_last=False)
        pretrain_idx_data_loader = get_idx_data_loader(mask.train_addr, batch_size=args.pretrain_batch_size, shuffle=True, drop_last=True)
        prevalid_idx_data_loader = get_idx_data_loader(mask.val_addr, batch_size=args.pretrain_batch_size, shuffle=False, drop_last=True)
        all_labels = full_data.addr_classes
    
    elif args.task == 'tx classification':
        tx_classifier = MLPClassifier(input_dim=args.hid_dim, dropout=args.dropout)
        model = nn.Sequential(dynamic_backbone, tx_classifier)
        train_idx_data_loader = get_idx_data_loader(mask.train_tx, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_idx_data_loader = get_idx_data_loader(mask.val_tx, batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_idx_data_loader = get_idx_data_loader(mask.test_tx, batch_size=args.batch_size, shuffle=False, drop_last=False)
        all_labels = full_data.tx_classes

        pretrain_idx_data_loader = get_idx_data_loader(mask.train_tx, batch_size=args.pretrain_batch_size, shuffle=True, drop_last=True)
        prevalid_idx_data_loader = get_idx_data_loader(mask.val_tx, batch_size=args.pretrain_batch_size, shuffle=False, drop_last=True)

    else:
        raise NotImplementedError(f'Invalid task name: {args.task}')

    print(f'model -> {model}')
    print(f'model name: {args.tgnn}, #parameters: {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

    if args.pretrain:
        assert args.tgnn == 'BitGAT'
        pretrain_tgnn(args, dynamic_backbone, full_data, pretrain_idx_data_loader, train_hgraph, prevalid_idx_data_loader, valid_hgraph)

    if args.freeze:
        for param in dynamic_backbone.parameters():
            param.requires_grad = False

    optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.lr, weight_decay=args.w_decay)

    model = convert_to_gpu(model, device=args.device)

    loss_func = nn.BCELoss()

    best_val_auc, early_stop = 0.0, 0
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()

        train_y_trues, train_y_preds, train_loss = inference(args, full_data, model, all_labels, optimizer, loss_func, epoch, train_idx_data_loader, train_hgraph, mode='train')
        train_metrics = Eval_Metrics(y_true=train_y_trues, y_pred_logit=train_y_preds)

        # eval
        model.eval()

        val_y_trues, val_y_preds, _ = inference(args, full_data, model, all_labels, optimizer, loss_func, epoch, val_idx_data_loader, valid_hgraph, mode='val')
        val_metrics = Eval_Metrics(y_true=val_y_trues, y_pred_logit=val_y_preds)

        print(f'Epoch: {epoch + 1}, train loss: {train_loss:.4f}, train auc: {train_metrics.auc:.4f}, val auc: {val_metrics.auc:.4f}')

        if val_metrics.auc > best_val_auc:
            best_val_auc = val_metrics.auc
            early_stop = 0
            best_model_state = deepcopy(model.state_dict())

        else:
            early_stop += 1

        if early_stop > args.patience:
            break
  
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.save(best_model_state, os.path.join(args.model_path, f'fold_{fold_seed}.pt'))
    model.load_state_dict(best_model_state)

    test_y_trues, test_y_preds, _ = inference(args, full_data, model, all_labels, optimizer, loss_func, epoch, test_idx_data_loader, test_hgraph, mode='test')
    test_metrics = Eval_Metrics(y_true=test_y_trues, y_pred_logit=test_y_preds)
    print(f'Fold {fold_seed + 1}, test auc: {test_metrics.auc:.4f}, test f1: {test_metrics.f1:.4f}, test ap: {test_metrics.ap:.4f}, test fpr95: {test_metrics.fpr95:.4f}')

    return test_metrics


@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


def inference(args, full_data, model, all_labels, optimizer, loss_func, epoch, idx_data_loader, hgraph, mode='train'):
    assert mode in ['train', 'val', 'test']
    total_loss, y_trues, y_preds = 0.0, [], []
    
    tbar = tqdm(idx_data_loader, ncols=180)

    with conditional_no_grad(mode in ['val', 'test']):
        for batch_idx, data_indices in enumerate(tbar):
            labels, predicts, loss = calculate_batch_loss(args, full_data, model, all_labels, loss_func, data_indices, hgraph)
            tbar.set_description(f'Epoch: {epoch + 1}, {mode} for the {batch_idx + 1} batch, {mode} batch loss: {loss.item():.4f}')
            total_loss += loss.item()
            y_trues.append(labels)
            y_preds.append(predicts)

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    total_loss /= (batch_idx + 1)
    return torch.cat(y_trues, dim=0).detach().cpu().numpy(), torch.cat(y_preds, dim=0).detach().cpu().numpy(), total_loss


def calculate_batch_loss(args, full_data, model, all_labels, loss_func, data_indices, hgraph):
    data_indices = data_indices.numpy()
    batch_labels = all_labels[data_indices]

    if args.tgnn == 'BitGAT':
        if args.task == 'addr classification':
            sub_hgraph = hgraph.sample_node(data_indices)
        else:
            sub_hgraph = hgraph.sample_edge(data_indices)
        batch_node_embeddings = model[0](sub_hgraph, task=args.task)

    elif args.tgnn in ['gcn', 'gpr', 'gat', 'gin', 'appnp', 'bwgnn']:
        if args.task == 'addr classification':
            g, node_indices = hgraph.sample_node2graph(data_indices)
            x = full_data.addr_attrs[node_indices]
        elif args.task == 'tx classification':
            g, edge_indices = hgraph.sample_edge2graph(data_indices)
            x = full_data.tx_attrs[edge_indices]
        
        g = g.to(args.device)
        x = torch.from_numpy(x).float().to(args.device)
        
        batch_node_embeddings = model[0](x, g)
    
    elif args.tgnn in ['HGNN', 'HGNN+', 'HNHN', 'UniGAT']:
        if args.task == 'tx classification':
            hg, edge_indices = hgraph.sample_edge2hypergraph(data_indices)
            x = full_data.tx_attrs[edge_indices]
        elif args.task == 'addr classification':
            hg, node_indices = hgraph.sample_node2hypergraph(data_indices)
            x = full_data.addr_attrs[node_indices]

        hg = hg.to(args.device)
        x = torch.from_numpy(x).float().to(args.device)
        batch_node_embeddings = model[0](x, hg)

    else:
        raise ValueError(f'Invalid model name: {args.tgnn}')

    predicts = model[1](batch_node_embeddings).squeeze(dim=-1).sigmoid()
    labels = torch.from_numpy(batch_labels).float().to(predicts.device)

    loss = loss_func(input=predicts, target=labels)

    return labels, predicts, loss


