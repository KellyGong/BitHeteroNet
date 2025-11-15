import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl.function as fn
from dhg.nn import HNHNConv


class HNHN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj):
        super(HNHN, self).__init__()

        self.inlinear = nn.Linear(in_channels, hidden_channels)
        self.outlinear = nn.Linear(hidden_channels, out_channels)
        
        torch.nn.init.xavier_uniform_(self.inlinear.weight)
        torch.nn.init.xavier_uniform_(self.outlinear.weight)

        self.layers = nn.ModuleList([HNHNConv(in_channels=hidden_channels,
                                              out_channels=hidden_channels,
                                              drop_rate=dropout) for _ in range(num_layers)])
        self.act = nn.LeakyReLU()

    def forward(self, x, hg, return_emb=False):
        x = self.inlinear(x)
        for conv in self.layers:
            x = conv(x, hg)
            x = self.act(x)
        if return_emb:
            x_out = self.outlinear(x)
            return x, x_out
        else:
            x = self.outlinear(x)
            return x
