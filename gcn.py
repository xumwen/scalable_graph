import math
import torch
import torch_geometric
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.data import Data, Batch, DataLoader, NeighborSampler, ClusterData, ClusterLoader


class MyGATConv(PyG.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels=1, **kwargs):
        super(MyGATConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.weight_n = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_e = nn.Parameter(torch.Tensor(edge_channels, out_channels))

        self.u = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.v = nn.Parameter(torch.Tensor(out_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_n)
        nn.init.xavier_uniform_(self.weight_e)
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        x = (
            torch.matmul(x[0], self.weight_n),
            torch.matmul(x[1], self.weight_n),
        )

        if len(edge_weight.shape) == 1:
            edge_weight = edge_weight.unsqueeze(dim=-1)

        edge_weight = torch.matmul(edge_weight, self.weight_e)

        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

    def message(self, edge_index_i, x_i, x_j, edge_weight):
        x_i = torch.matmul(x_i, self.u)
        x_j = torch.matmul(x_j, self.v)

        gate = F.sigmoid(x_i * x_j * edge_weight.unsqueeze(dim=1))

        return x_j * gate

    def update(self, aggr_out, x):
        x = x[1]
        aggr_out = torch.matmul(x, self.u) + aggr_out
        aggr_out = aggr_out.permute(1, 0, 2)
        bn = nn.BatchNorm1d(aggr_out.shape[1]).to(x.device)
        aggr_out = bn(aggr_out)
        aggr_out = aggr_out.permute(1, 0, 2)

        return x + aggr_out


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = MyGATConv(in_channels=in_channels, out_channels=16)
        self.conv2 = MyGATConv(in_channels=16, out_channels=out_channels)

    def forward(self, X, g):
        edge_index = g['edge_index']
        edge_weight = g['edge_weight']

        size = g['size']
        res_n_id = g['res_n_id']

        # swap node to dim 0
        X = X.permute(1, 0, 2)

        conv1 = self.conv1(
            (X, X[res_n_id[0]]), edge_index[0], edge_weight=edge_weight[0], size=size[0]
        )

        X = F.leaky_relu(conv1)

        conv2 = self.conv2(
            (X, X[res_n_id[1]]), edge_index[1], edge_weight=edge_weight[1], size=size[1]
        )

        X = F.leaky_relu(conv2)

        X = X.permute(1, 0, 2)
        return X


class MySAGEConv(PyG.SAGEConv):
    def __init__(self, in_channels, out_channels, normalize=False, concat=False, bias=True, **kwargs):
        super(MySAGEConv, self).__init__(in_channels, out_channels,
                                         normalize=normalize, concat=concat, bias=bias, **kwargs)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1, 1) * x_j

    def update(self, aggr_out, x, res_n_id):
        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)
        elif self.concat and (isinstance(x, tuple) or isinstance(x, list)):
            assert res_n_id is not None
            # TODO: to check the consistency
            aggr_out = torch.cat([x[0][res_n_id], aggr_out], dim=-1)

        aggr_out = torch.matmul(aggr_out, self.weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out


class SAGENet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGENet, self).__init__()
        self.conv1 = MySAGEConv(
            in_channels, 16, normalize=False, concat=True)
        self.conv2 = MySAGEConv(
            16, out_channels, normalize=False, concat=True)

    def forward(self, X, g):
        edge_index = g['edge_index']
        edge_weight = g['edge_weight']

        size = g['size']
        res_n_id = g['res_n_id']

        # swap node to dim 0
        X = X.permute(1, 0, 2)

        conv1 = self.conv1(
            (X, None), edge_index[0], edge_weight=edge_weight[0], size=size[0], res_n_id=res_n_id[0])

        X = F.leaky_relu(conv1)

        conv2 = self.conv2(
            (X, None), edge_index[1], edge_weight=edge_weight[1], size=size[1], res_n_id=res_n_id[1])

        X = F.leaky_relu(conv2)
        X = X.permute(1, 0, 2)
        return X
