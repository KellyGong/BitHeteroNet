import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj):
        super(GCN, self).__init__()
        self.gnn = GCN_sparse(in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj)

    def forward(self, x, g=None):
        return self.gnn.forward(x=x, g=g)


class GCN_sparse(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj):
        super(GCN_sparse, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
        self.layers.append(GCNConv_dgl(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

    def forward(self, x, g=None):
        assert g is not None
        g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, g)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, g)
        return x


class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']
