import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import GCNBlock

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                num_nodes, gcn_type, normalize):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=spatial_channels)
        self.gcn = GCNBlock(in_channels=spatial_channels,
                            spatial_channels=spatial_channels,
                            num_nodes=num_nodes,
                            gcn_type=gcn_type,
                            normalize=normalize
                            )
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)

    def forward(self, X, g):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A: Adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes, num_timesteps_out,
        num_features=out_channels).
        """
        t1 = self.temporal1(X)
        gcn_out = self.gcn(t1, g)
        t2 = self.temporal2(gcn_out)
        
        return t2, gcn_out


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, config):
        super(STGCN, self).__init__()
        num_nodes = getattr(config, 'num_nodes')
        num_features = getattr(config, 'num_features')
        num_timesteps_input = getattr(config, 'num_timesteps_input')
        num_timesteps_output = getattr(config, 'num_timesteps_output')

        hidden_size = getattr(config, 'hidden_size', 64)
        gcn_type = getattr(config, 'gcn', 'gat')
        normalize = getattr(config, 'normalize', 'none')

        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=hidden_size, num_nodes=num_nodes,
                                 gcn_type=gcn_type, normalize=normalize)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=hidden_size, num_nodes=num_nodes,
                                 gcn_type=gcn_type, normalize=normalize)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

    def forward(self, X, g):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A: Normalized adjacency matrix.
        """
        # stgcn only support subgraph training
        assert(g['type'] == 'subgraph')

        out1, _ = self.block1(X, g)
        out2, gcn_out = self.block2(out1, g)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4, gcn_out