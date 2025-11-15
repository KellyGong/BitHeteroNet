import os
import torch
import random
import argparse
import os.path as osp
from utils import Eval_Metrics_Average
from dataloader import load_all_data, get_addr_classification_data, get_tx_classification_data
from train_tgnn import train_eval_tgnn


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_eval(args, full_data, mask):
    eval_metrics_avg = Eval_Metrics_Average()
    for fold_seed in range(args.fold):
        eval_metrics = train_eval_tgnn(args, full_data, mask, fold_seed)
        eval_metrics_avg(eval_metrics)
    print(eval_metrics_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # task setting
    parser.add_argument('--task', type=str, default='tx classification', choices=['addr classification', 'tx classification'])
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--transduction', type=str2bool, default='false', help='Boolean value for transductive setting for addr classification task.')
    parser.add_argument('--time_split', type=str2bool, default='true', help='Boolean value for time-based split for tx classification task.')
    parser.add_argument('--fold', type=int, default=5, help='Fold number')

    # model types
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tgnn', type=str, default='BitGAT', choices=['BitGAT', 
                                                                       'gcn', 'gpr', 'gat', 'gin', 'appnp', 'bwgnn',
                                                                       'HGNN', 'HGNN+', 'HNHN', 'UniGAT'])

    # training setting
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'])

    # gnn
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dropout_adj', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--w_decay', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=20)

    # tgnn
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden dimension of edge and node features')
    parser.add_argument('--time_feat_dim', type=int, default=16)
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')

    # pretrain
    parser.add_argument('--pretrain', type=str2bool, default='false')
    parser.add_argument('--pretrain_epochs', type=int, default=20)
    parser.add_argument('--pretrain_lr', type=float, default=0.005)
    parser.add_argument('--pretrain_w_decay', type=float, default=0.00001)
    parser.add_argument('--pretrain_rate', type=float, default=0.1)
    parser.add_argument('--pretrain_batch_size', type=int, default=100)
    parser.add_argument('--freeze', type=str2bool, default='true')
    parser.add_argument('--aug1', type=str, default='nodedrop', choices=['identity', 'featmask', 'featdist', 'edgedrop', 'nodedrop', 'incidrop'])
    parser.add_argument('--aug2', type=str, default='edgedrop', choices=['identity', 'featmask', 'featdist', 'edgedrop', 'nodedrop', 'incidrop'])

    args = parser.parse_args()

    full_data = load_all_data()

    print(f'task: {args.task}')

    model_id = random.randint(0, 100000)
    if args.task == 'addr classification':
        model_name = f'{args.tgnn}_trans.pth' if args.transduction else f'{args.tgnn}_ind_{str(model_id)}'
        args.model_path = osp.join('best_model', f'{args.task}', f'{model_name}')
        mask = get_addr_classification_data(full_data, train_ratio=args.train_ratio, transductive=args.transduction)
    elif args.task == 'tx classification':
        model_name = f'{args.tgnn}_time.pth' if args.time_split else f'{args.tgnn}_ind_{str(model_id)}'
        args.model_path = osp.join('best_model', f'{args.task}', f'{model_name}')
        mask = get_tx_classification_data(full_data, train_ratio=args.train_ratio, time_split=args.time_split)
    else:
        raise ValueError('Invalid task')

    train_eval(args, full_data, mask)
