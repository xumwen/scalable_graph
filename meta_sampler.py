import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import IterableDataset
from torch.distributions import Normal
from torch_geometric.data import Data

import numpy as np
import pandas as pd

from ppo import PPO, ActorDemo


class MetaSampler(object):
    def __init__(self, num_nodes, node_emb, edge_index, subgraph_nodes, sample_step=2, shuffle=False, random_sample=False):
        self.num_nodes = num_nodes
        self.node_emb = node_emb
        self.edge_index = edge_index
        self.subgraph_nodes = subgraph_nodes
        self.sample_step = sample_step
        self.shuffle = shuffle
        self.random_sample = random_sample

        self.state_size = node_emb.shape[1] * 2
        self.model = ActorDemo(state_size=self.state_size)
        # self.model = PPO()
    
    def __get_init_nodes__(self, num_init_nodes=10):
        left_nodes = np.where(self.node_visit == 0)[0]
        if self.shuffle:
            np.random.shuffle(left_nodes)
        if len(left_nodes) <= num_init_nodes:
            return left_nodes
            
        return left_nodes[:num_init_nodes]

    def __produce_subgraph__(self, n_id, subgraph_nodes):
        row, col = self.edge_index
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        edge_mask = torch.zeros(row.size(0), dtype=torch.bool)
        
        for i in range(self.sample_step):
            # get 1-hop neighbor
            node_mask[n_id] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            new_n_id = col[edge_mask].numpy()
            
            # remove visited and cent nodes
            tmp_node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            tmp_node_mask[new_n_id] = True
            neighbor_id = np.where(tmp_node_mask & (node_mask==False) & (self.node_visit==False))[0]

            if self.random_sample:
                # random sample to warm start without node_emb
                if subgraph_nodes >= self.num_nodes - self.node_visit.sum():
                    sample_n_id = np.where(self.node_visit == False)[0]
                else:
                    num_left = subgraph_nodes - len(n_id)
                    if len(neighbor_id) >= num_left:
                        np.random.shuffle(neighbor_id)
                        sample_n_id = neighbor_id[:num_left]
                    else:
                        sample_n_id = neighbor_id
            else:
                # use ppo to take action
                cent_emb = self.node_emb[n_id].sum(dim=0)
                neighbor_emb = self.node_emb[neighbor_id].sum(dim=0)
                s = torch.cat([cent_emb, neighbor_emb], dim=0)
                mu1, sigma1, prob = self.model.action(s)

                # calculate kl-divergense to sample nodes
                mu2 = self.node_emb[neighbor_id].mean(dim=1)
                sigma2 = self.node_emb[neighbor_id].std(dim=1)
                kl_div = (sigma2 / sigma1).log() + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
                weight = torch.exp(-kl_div)
                sample_n_id = neighbor_id[torch.bernoulli(weight) == 1]

            n_id = np.union1d(n_id, sample_n_id)
            if len(n_id) >= subgraph_nodes:
                break

        self.node_visit[n_id] = True
        # return subgraph
        node_mask[n_id] = True
        edge_mask = node_mask[row] & node_mask[col]
        e_id = np.where(edge_mask)[0]
        # edge_index reindex by n_id
        tmp = torch.empty(self.num_nodes, dtype=torch.long)
        tmp[n_id] = torch.arange(len(n_id))
        edge_index = tmp[self.edge_index[:, e_id]]

        return Data(edge_index=edge_index, n_id=n_id, e_id=e_id)

    def __call__(self):
        r"""Returns a generator of subgraph"""
        self.node_visit = torch.zeros(self.num_nodes, dtype=torch.bool)
        produce = self.__produce_subgraph__
        while self.node_visit.sum() != self.num_nodes:
            n_id =  self.__get_init_nodes__()
            yield produce(n_id, self.subgraph_nodes)


class MetaSamplerDataset(IterableDataset):
    def __init__(self, X, y, num_nodes, node_emb, edge_index, edge_weight, batch_size, subgraph_nodes=200, shuffle=False, random_sample=False):
        self.X = X
        self.y = y
        self.node_emb = node_emb

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.subgraph_nodes = subgraph_nodes
        self.shuffle = shuffle
        self.random_sample = random_sample

        self.meta_sampler = MetaSampler(num_nodes, node_emb, edge_index, subgraph_nodes, shuffle=shuffle, random_sample=random_sample)
        self.length = self.get_length()

    def get_subgraph(self, subgraph):
        g = {
            'edge_index': subgraph.edge_index,
            'edge_weight': self.edge_weight[subgraph.e_id],
            'n_id': subgraph.n_id,
            'e_id': subgraph.e_id
        }

        return g

    def __iter__(self):
        for subgraph in self.meta_sampler():
            g = self.get_subgraph(subgraph)
            X, y = self.X[:, g['n_id']], self.y[:, g['n_id']]
            dataset_len = X.size(0)
            indices = list(range(dataset_len))

            if self.shuffle:
                np.random.shuffle(indices)

            num_batches = (len(indices) + self.batch_size -
                        1) // self.batch_size
            for batch_id in range(num_batches):
                start = batch_id * self.batch_size
                end = (batch_id + 1) * self.batch_size
                yield X[indices[start: end]], y[indices[start: end]], g, torch.LongTensor(indices[start: end])

    def get_length(self):
        length = 0

        for _ in self.meta_sampler():
            num_samples_per_node = self.X.size(0)
            length += (num_samples_per_node +
                    self.batch_size - 1) // self.batch_size

        return length

    def __len__(self):
        return self.length