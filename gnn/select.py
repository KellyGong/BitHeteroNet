from .gcn import GCN
from .gpr import GPR
from .gat import GAT
from .gin import GIN
from .appnp import APPNP
from .bwgnn import BWGNN
from .hgnn import HGNN
from .hgnn_plus import HGNN_PLUS
from .hnhn import HNHN
from .unigat import UNIGAT


def get_gnn_model(model_str, in_channels, hidden_channels, out_channels, num_layers,
                  dropout, dropout_adj):
    if model_str == 'gcn':
        Model = GCN
    elif model_str == 'gpr':
        Model = GPR
    elif model_str == 'gat':
        Model = GAT
    elif model_str == 'gin':
        Model = GIN
    elif model_str == 'appnp':
        Model = APPNP
    elif model_str == 'bwgnn':
        Model = BWGNN
    elif model_str == 'HGNN':
        Model = HGNN
    elif model_str == 'HGNN+':
        Model = HGNN_PLUS
    elif model_str == 'HNHN':
        Model = HNHN
    elif model_str == 'UniGAT':
        Model = UNIGAT
    else:
        raise NotImplementedError

    return Model(in_channels=in_channels,
                 hidden_channels=hidden_channels,
                 out_channels=out_channels,
                 num_layers=num_layers,
                 dropout=dropout,
                 dropout_adj=dropout_adj)
