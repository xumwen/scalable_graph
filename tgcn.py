import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import SAGENet, GATNet, GatedGCNNet, EGNNNet
from krnn import KRNN
from gru import GRUBlock

from torch_geometric.data import Data, Batch, DataLoader, NeighborSampler, ClusterData, ClusterLoader


class GCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, num_nodes, gcn_type, normalize):
        super(GCNBlock, self).__init__()
        GCNUnit = {'sage': SAGENet, 'gat': GATNet,
                   'gated': GatedGCNNet, 'egnn': EGNNNet}.get(gcn_type)
        self.gcn = GCNUnit(in_channels=in_channels,
                           out_channels=spatial_channels, 
                           normalize=normalize)

    def forward(self, X, g):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param g: graph information.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t1 = X.permute(0, 2, 1, 3).contiguous(
        ).view(-1, X.shape[1], X.shape[3])
        t2 = F.relu(self.gcn(t1, g))
        out = t2.view(X.shape[0], X.shape[2], t2.shape[1],
                      t2.shape[2]).permute(0, 2, 1, 3)

        return out


class TGCN(nn.Module):
    def __init__(self, config):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(TGCN, self).__init__()

        num_nodes = getattr(config, 'num_nodes')
        num_features = getattr(config, 'num_features')
        num_timesteps_input = getattr(config, 'num_timesteps_input')
        num_timesteps_output = getattr(config, 'num_timesteps_output')

        hidden_size = getattr(config, 'hidden_size', 64)
        gcn_type = getattr(config, 'gcn', 'gat')
        self.rnn_type = getattr(config, 'rnn', 'gru')
        normalize = getattr(config, 'normalize', 'none')

        self.gcn = GCNBlock(in_channels=num_features,
                            spatial_channels=hidden_size,
                            num_nodes=num_nodes,
                            gcn_type=gcn_type,
                            normalize=normalize
                            )

        if self.rnn_type == 'gru':
            self.gru = GRUBlock(hidden_size, hidden_size, num_timesteps_input,
                                num_timesteps_output)
        else:
            self.krnn = KRNN(num_nodes, hidden_size, num_timesteps_input,
                            num_timesteps_output, hidden_size)

    def forward(self, X, g):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        gcn_out = self.gcn(X, g)
        if self.rnn_type == 'gru':
            rnn_out = self.gru(gcn_out)
        else:
            _, rnn_out = self.krnn(gcn_out, g['cent_n_id'])

        return rnn_out, gcn_out
