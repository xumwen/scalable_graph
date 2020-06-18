import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import IterableDataset
from torch.distributions import Normal
from torch_geometric.data import Data

import numpy as np
import pandas as pd


class MetaSampler(object):
    def __init__(self, policy, num_nodes, node_emb, edge_index, subgraph_nodes, sample_step=2, shuffle=False, random_sample=False):
        self.policy = policy
        self.num_nodes = num_nodes
        self.node_emb = node_emb
        self.edge_index = edge_index
        self.subgraph_nodes = subgraph_nodes
        self.sample_step = sample_step
        self.shuffle = shuffle
        self.random_sample = random_sample
        self.subgraphs = []
        self.reset()
    
    def reset(self):
        self.node_visit = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.subgraphs = []
    
    def get_init_nodes(self, num_init_nodes=10):
        left_nodes = np.where(self.node_visit == 0)[0]
        if self.shuffle:
            np.random.shuffle(left_nodes)
        if len(left_nodes) <= num_init_nodes:
            return left_nodes
            
        return left_nodes[:num_init_nodes]

    def get_neighbor(self, n_id):
        row, col = self.edge_index
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        edge_mask = torch.zeros(row.size(0), dtype=torch.bool)

        # get 1-hop neighbor
        node_mask[n_id] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_n_id = col[edge_mask].numpy()
        
        # remove visited and cent nodes
        tmp_node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        tmp_node_mask[new_n_id] = True
        neighbor_id = np.where(tmp_node_mask & (node_mask==False) & (self.node_visit==False))[0]

        return neighbor_id

    def random_sample_left_nodes(self, n_id, neighbor_id):
        """ 
        Random sample to warm start without node_emb
        and to avoid exceeding subgraph_nodes
        """
        num_left = self.subgraph_nodes - len(n_id)
        if len(neighbor_id) >= num_left:
            np.random.shuffle(neighbor_id)
            sample_n_id = neighbor_id[:num_left]
        else:
            sample_n_id = neighbor_id
        
        return sample_n_id

    def neighbor_sample(self, neighbor_id, action):
        """ 
        Neighbor sample calculate kl-divergence between
        node distribution and action distribution
        """

        # calculate kl-divergense to sample nodes
        action = self.rescale_action(action)
        mu1, sigma1 = action[0], action[1]
        mu2 = self.node_emb[neighbor_id].mean(dim=1)
        sigma2 = self.node_emb[neighbor_id].std(dim=1)

        kl_div = (sigma2 / sigma1).log() + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
        weight = torch.exp(-kl_div)
        sample_n_id = neighbor_id[torch.bernoulli(weight) == 1]

        return sample_n_id
    
    def get_state(self, n_id, neighbor_id):
        cent_emb = self.node_emb[n_id].sum(dim=0)
        neighbor_emb = self.node_emb[neighbor_id].sum(dim=0)

        state = torch.cat([cent_emb, neighbor_emb], dim=0)
        return state
    
    def rescale_action(self, action):
        # rescale action to get a sample distribution close to nodes_emb distribution
        mu, sigma = action[0], action[1]

        nodes_mu = self.node_emb.mean(dim=1)
        mu_max = nodes_mu.max()
        mu_min = nodes_mu.min()

        nodes_sigma = self.node_emb.std(dim=1)
        sigma_max = nodes_sigma.max()
        sigma_min = nodes_sigma.min()

        mu = np.tanh(mu) * (mu_max - mu_min) + (mu_max + mu_min) / 2
        sigma = np.tanh(sigma) * (sigma_max - sigma_min) / 2 + (sigma_max + sigma_min) / 2

        return [mu, sigma]


    def __produce_subgraph_by_nodes__(self, n_id):
        row, col = self.edge_index
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        edge_mask = torch.zeros(row.size(0), dtype=torch.bool)

        # visited nodes update
        self.node_visit[n_id] = True

        # get edge
        node_mask[n_id] = True
        edge_mask = node_mask[row] & node_mask[col]
        e_id = np.where(edge_mask)[0]

        # edge_index reindex by n_id
        tmp = torch.empty(self.num_nodes, dtype=torch.long)
        tmp[n_id] = torch.arange(len(n_id))
        edge_index = tmp[self.edge_index[:, e_id]]

        return Data(edge_index=edge_index, n_id=n_id, e_id=e_id)

    def __produce_subgraph__(self):
        if self.num_nodes - self.node_visit.sum() <= self.subgraph_nodes:
            # last subgraph
            n_id = np.where(self.node_visit == False)[0]
            return self.__produce_subgraph_by_nodes__(n_id)

        # sample several steps
        n_id = self.get_init_nodes()

        for i in range(self.sample_step):
            neighbor_id = self.get_neighbor(n_id)
            if self.random_sample:
                sample_n_id = self.random_sample_left_nodes(n_id, neighbor_id)
            else:
                s = self.get_state(n_id, neighbor_id)
                action, logp = self.policy.action(s)
                sample_n_id = self.neighbor_sample(neighbor_id, action)
            
            n_id = np.union1d(n_id, sample_n_id)
            if len(n_id) >= self.subgraph_nodes:
                break

        return self.__produce_subgraph_by_nodes__(n_id)

    def __call__(self):
        self.reset()
        while self.node_visit.sum() != self.num_nodes:
            self.subgraphs.append(self.__produce_subgraph__())


class MetaSamplerDataset(IterableDataset):
    def __init__(self, X, y, meta_sampler, num_nodes, edge_index, 
                edge_weight, batch_size, shuffle=False):
        self.X = X
        self.y = y
        self.meta_sampler = meta_sampler

        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.sample_subgraph()
        self.length = self.get_length()

    def sample_subgraph(self):
        if len(self.meta_sampler.subgraphs) == 0:
            self.meta_sampler()

    def get_subgraph(self, subgraph):
        g = {
            'edge_index': subgraph.edge_index,
            'edge_weight': self.edge_weight[subgraph.e_id],
            'n_id': subgraph.n_id,
            'e_id': subgraph.e_id
        }

        return g

    def __iter__(self):
        # need pre-sampled subgraphs to avoid length inconsistency
        for subgraph in self.meta_sampler.subgraphs:
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

        for _ in self.meta_sampler.subgraphs:
            num_samples_per_node = self.X.size(0)
            length += (num_samples_per_node +
                    self.batch_size - 1) // self.batch_size

        return length

    def __len__(self):
        return self.length