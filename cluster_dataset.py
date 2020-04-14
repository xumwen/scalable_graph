import torch 
import numpy as np

from torch_geometric.data import Data, ClusterData, ClusterLoader
from torch.utils.data import DataLoader, TensorDataset, IterableDataset

class ClusterDataset(IterableDataset):
    def __init__(self, X, y, edge_index, edge_weight, num_nodes, batch_size, shuffle):
        self.X = X
        self.y = y

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.graph_cluster = self.make_graph_cluster()

        self.length = self.get_length()

    def make_graph_cluster(self):
        graph = Data(
            edge_index=self.edge_index, edge_attr=self.edge_weight, 
            n_id=torch.arange(0, self.num_nodes), num_nodes=self.num_nodes
        ).to('cpu')

        cluster_data = ClusterData(
            graph, num_parts=50, recursive=False, save_dir='./data'
        )

        cluster_loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True, num_workers=0)

        return cluster_loader

    def get_subgraph(self, subgraph):
        graph = dict()

        device = self.edge_index.device

        graph['edge_index'] = subgraph.edge_index.to(device)
        graph['edge_weight'] = subgraph.edge_attr.to(device)
        graph['cent_n_id'] = subgraph.n_id

        return graph

    def __iter__(self):
        for subgraph in self.graph_cluster:
            g = self.get_subgraph(subgraph)
            X, y = self.X[:, g['cent_n_id']], self.y[:, g['cent_n_id']]

            subset = TensorDataset(X, y)
            indices = np.arange(len(subset))

            if self.shuffle:
                np.random.shuffle(indices)

            num_batches = (len(subset) + self.batch_size -
                           1) // self.batch_size

            for batch_id in range(num_batches):
                start = batch_id * self.batch_size
                end = (batch_id + 1) * self.batch_size
                yield X[indices[start: end]], y[indices[start: end]], g

    def get_length(self):
        length = 0
        for subgraph in self.graph_cluster:
            length += (self.y.size(0) + self.batch_size - 1) // self.batch_size
        return length

    def __len__(self):
        return self.length