import os
import time
import argparse
import json
import math
import torch
import torch_geometric
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from argparse import Namespace
from torch_geometric.data import Data, Batch, NeighborSampler, ClusterData, ClusterLoader
import torch.nn as nn
import torch.distributed as dist
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tgcn import TGCN
from stgcn import STGCN
from sandwich import Sandwich

from preprocess import generate_dataset, load_nyc_sharing_bike_data, load_metr_la_data, load_pems_d7_data, load_pems_m_data, get_normalized_adj
from base_task import add_config_to_argparse, BaseConfig, BasePytorchTask, \
    LOSS_KEY, BAR_KEY, SCALAR_LOG_KEY, VAL_SCORE_KEY
from dataset import NeighborSampleDataset, ClusterDataset, SAINTDataset
from meta_sampler import MetaSampler, MetaSamplerDataset
from ppo import PPO, MetaSampleEnv


class STConfig(BaseConfig):
    def __init__(self):
        super(STConfig, self).__init__()
        # 1. Reset base config variables:
        self.max_epochs = 100
        self.early_stop_epochs = 20

        # 2. set spatial-temporal config variables:
        self.model = 'stgcn'  # choices: tgcn, stgcn, sandwich
        self.dataset = 'nyc'  # choices: metr, nyc, pems, pems-m
        # choices: ./data/METR-LA, ./data/NYC-Sharing-Bike
        self.data_dir = './data/NYC-Sharing-Bike'
        self.gcn = 'gat'  # choices: sage, gat, egnn
        self.rnn = 'gru'   # choices: gru, krnn

        # per-gpu training batch size, real_batch_size = batch_size * num_gpus * grad_accum_steps
        self.batch_size = 32
        self.hidden_size = 64 # node embedding size
        self.normalize = 'none' # normalize after gnn
        self.num_timesteps_input = 12  # the length of the input time-series sequence
        self.num_timesteps_output = 3  # the length of the output time-series sequence
        self.lr = 1e-3  # the learning rate

        # pretrained ckpt for krnn, use 'none' to ignore it
        self.pretrain_ckpt = 'none'
        self.use_residual = False

        # graph sampling choices
        self.graph_sampling = 'meta'  # choices: neighbor, cluster, saint, meta
        # for neighbor sampling
        self.neighbor_sampling_size = 5  # the number of neighbors to be sampled at each layer
        self.neighbor_batch_size = 100  # the number of central nodes to be expanded at each batch
        # for cluster gcn
        self.cluster_part_num = 100  # the number of partitions to divide the graph
        self.cluster_batch_size = 10  # the number of partitions to load at each batch
        # for graph saint
        self.saint_sample_type = 'random_walk'  # choices: random_walk, node
        self.saint_batch_size = 100  # the number of initial nodes for random walk at each batch
        self.saint_walk_length = 2  # the length of each random walk
        self.saint_sample_coverage = 50  # the number of counts per node to computation norm statistics
        self.use_saint_norm = True  # whether to use SAINT normalization tricks
        # for meta sample
        self.subgraph_nodes = 70 # num nodes of subgraph produced by meta sampler
        self.moving_avg = 0.9 # update node embedding with moving average to avoid value accumulation
        self.state_size = self.hidden_size * 2 # state_size of ppo


def get_model_class(model):
    return {
        'tgcn': TGCN,
        'stgcn': STGCN,
        'sandwich': Sandwich
    }.get(model)


class WrapperNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_class = get_model_class(config.model)
        self.net = model_class(config)

    def forward(self, X, g):
        return self.net(X, g)


class SpatialTemporalTask(BasePytorchTask):
    def __init__(self, config):
        super(SpatialTemporalTask, self).__init__(config)
        self.log('Intialize {}'.format(self.__class__))

        self.init_data()
        self.loss_func = nn.MSELoss()

        self.log('Config:\n{}'.format(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=4)
        ))
    
    def init_ppo(self):
        self.ppo = PPO(state_size=self.config.state_size, device=self.device)
        self.node_emb = torch.zeros(self.config.num_nodes, self.config.hidden_size)
        self.epoch_node_emb = torch.zeros(self.config.num_nodes, self.config.hidden_size)

    def init_data(self, data_dir=None):
        if data_dir is None:
            data_dir = self.config.data_dir

        if self.config.dataset == "metr":
            A, X, means, stds = load_metr_la_data(data_dir)
        elif self.config.dataset == "nyc":
            A, X, means, stds = load_nyc_sharing_bike_data(data_dir)
        elif self.config.dataset == "pems":
            A, X, means, stds = load_pems_d7_data(data_dir)
        elif self.config.dataset == "pems-m":
            A, X, means, stds = load_pems_m_data(data_dir)

        X, y = generate_dataset(X, 
            num_timesteps_input=self.config.num_timesteps_input, 
            num_timesteps_output=self.config.num_timesteps_output,
            dataset = self.config.dataset)

        split_line1 = int(X.shape[0] * 0.6)
        split_line2 = int(X.shape[0] * 0.8)

        self.training_input, self.training_target = X[:split_line1], y[:split_line1]
        self.val_input, self.val_target = X[split_line1:split_line2], y[split_line1:split_line2]
        self.test_input, self.test_target = X[split_line2:], y[split_line2:]

        self.A = torch.from_numpy(A)
        self.sparse_A = self.A.to_sparse()
        self.edge_index = self.sparse_A._indices()
        self.edge_weight = self.sparse_A._values()

        contains_self_loops = torch_geometric.utils.contains_self_loops(
            self.edge_index)
        self.log('Contains self loops: {}, but we add them.'.format(
            contains_self_loops))
        if not contains_self_loops:
            self.edge_index, self.edge_weight = torch_geometric.utils.add_self_loops(
                self.edge_index, self.edge_weight,
                num_nodes=self.A.shape[0]
            )

        # set config attributes for model initialization
        self.config.num_nodes = self.A.shape[0]
        self.config.num_edges = self.edge_weight.shape[0]
        self.config.num_features = self.training_input.shape[3]
        self.log('Total nodes: {}'.format(self.config.num_nodes))
        self.log('Average degree: {:.3f}'.format(
            self.config.num_edges / self.config.num_nodes))
    
    def update_epoch_node_embedding(self, n_id, node_emb):
        mean_node_emb = node_emb.mean(dim=[0, 2]).to('cpu')
        self.epoch_node_emb[n_id] += mean_node_emb
    
    def update_node_embedding(self):
        # normalize current epoch embedding
        mean = self.epoch_node_emb.mean()
        std = self.epoch_node_emb.std()
        self.epoch_node_emb = (self.epoch_node_emb - mean) / (std + 1e-5)

        # update node embedding by moving average
        self.node_emb = self.config.moving_avg * self.node_emb + \
            (1 - self.config.moving_avg) * self.epoch_node_emb
        print("node_emb mean:", self.node_emb.mean().item())
        
        # reset epoch embedding
        self.epoch_node_emb = torch.zeros(self.config.num_nodes, self.config.hidden_size)

    def make_sample_dataloader(self, X, y, epoch, shuffle=True, use_dist_sampler=False, rep_eval=None):
        if self.config.graph_sampling == 'neighbor':
            dataset = NeighborSampleDataset(
                X, y, self.edge_index, self.edge_weight, self.config.num_nodes, self.config.batch_size,
                shuffle=shuffle, use_dist_sampler=use_dist_sampler, rep_eval=rep_eval,
                neighbor_sampling_size=self.config.neighbor_sampling_size,
                neighbor_batch_size=self.config.neighbor_batch_size)
        elif self.config.graph_sampling == 'cluster':
            dataset = ClusterDataset(
                X, y, self.edge_index, self.edge_weight, self.config.num_nodes, self.config.batch_size,
                shuffle=shuffle, use_dist_sampler=use_dist_sampler, rep_eval=rep_eval,
                cluster_part_num=self.config.cluster_part_num,
                cluster_batch_size=self.config.cluster_batch_size)
        elif self.config.graph_sampling == 'saint':
            dataset = SAINTDataset(
                X, y, self.edge_index, self.edge_weight, self.config.num_nodes, self.config.batch_size,
                shuffle=shuffle, use_dist_sampler=use_dist_sampler, rep_eval=rep_eval,
                saint_sample_type=self.config.saint_sample_type,
                saint_batch_size=self.config.saint_batch_size,
                saint_walk_length=self.config.saint_walk_length,
                saint_sample_coverage=self.config.saint_sample_coverage,
                use_saint_norm=self.config.use_saint_norm)
        elif self.config.graph_sampling == 'meta':
            meta_sampler = MetaSampler(
                self.ppo.policy, self.config.num_nodes, self.node_emb, self.edge_index, 
                self.config.subgraph_nodes, shuffle=shuffle, random_sample=(epoch==1)
            )

            # return a data loader based on meta sampler
            dataset = MetaSamplerDataset(
                X, y, meta_sampler, self.config.num_nodes, 
                self.edge_index, self.edge_weight, 
                self.config.batch_size, shuffle=shuffle,
                use_dist_sampler = use_dist_sampler
            )
        else:
            raise Exception('Unsupported graph sampling type: {}'.format(self.config.graph_sampling))

        return DataLoader(dataset, batch_size=None)

    def build_train_dataloader(self, epoch):
        return self.make_sample_dataloader(
            self.training_input, self.training_target, 
            epoch, use_dist_sampler=True
        )

    def build_val_dataloader(self, epoch):
        return self.make_sample_dataloader(
            self.val_input, self.val_target, 
            epoch, use_dist_sampler=False
        )

    def build_test_dataloader(self, epoch):
        return self.make_sample_dataloader(
            self.test_input, self.test_target, 
            epoch, use_dist_sampler=False
        )

    def build_optimizer(self, model):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def load_pretrain_ckpt(self):
        if self.config.pretrain_ckpt == 'none':
            return
        state_dict = torch.load(self.config.pretrain_ckpt)['model']

        if self.has_parallel_wrapper(self.model):
            model = self.model.module.net.gru1.seq2seq
        else:
            model = self.model.net.gru1.seq2seq

        for name, param in model.named_parameters():
            name = 'net.seq2seq.{}'.format(name)
            param.data.copy_(state_dict[name])
            # # if krnn is pretrained, we then freeze it
            param.requires_grad = False

    def train_step(self, batch, batch_idx):
        X, y, g, rows = batch

        y_hat, node_emb = self.model(X, g)
        assert(y.size() == y_hat.size())
        loss = self.loss_func(y_hat, y)
        loss_i = loss.item()  # scalar loss

        # update node embedding for meta sampler
        if self.config.graph_sampling == 'meta':
            self.update_epoch_node_embedding(g['cent_n_id'], node_emb)

        return {
            LOSS_KEY: loss,
            BAR_KEY: {'train_loss': loss_i},
            SCALAR_LOG_KEY: {'train_loss': loss_i}
        }

    def eval_step(self, batch, batch_idx, tag):
        X, y, g, rows = batch

        y_hat, node_emb = self.model(X, g)
        assert(y.size() == y_hat.size())

        loss = self.loss_func(y, y_hat)

        return {'loss': loss}

    def eval_epoch_end(self, outputs, tag):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        loss = loss.item()

        out = {
            BAR_KEY: {'{}_loss'.format(tag): loss},
            SCALAR_LOG_KEY: {'{}_loss'.format(tag): loss},
            VAL_SCORE_KEY: -loss,  # a larger score corresponds to a better model
        }

        return out

    def val_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def val_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')
    
    def train_ppo_step(self, model):
        self.update_node_embedding()
        env = MetaSampleEnv(
            model, self.val_input, self.val_target, 
            self.config.num_nodes, self.node_emb, 
            self.edge_index, self.edge_weight, 
            self.config.subgraph_nodes, self.config.batch_size,
            self.device
        )
        self.ppo.train_step(env)


if __name__ == '__main__':
    start_time = time.time()

    # build argument parser and config
    config = STConfig()
    parser = argparse.ArgumentParser(description='Spatial-Temporal-Task')
    add_config_to_argparse(config, parser)

    # parse arguments to config
    args = parser.parse_args()
    config.update_by_dict(args.__dict__)

    # build task
    task = SpatialTemporalTask(config)

    # Set random seed before the initialization of network parameters
    # Necessary for distributed training
    task.set_random_seed()
    net = WrapperNet(task.config)
    task.init_model_and_optimizer(net)
    task.load_pretrain_ckpt()

    if not task.config.skip_train:
        task.fit()

    # Resume the best checkpoint for evaluation
    task.resume_best_checkpoint()
    val_eval_out = task.val_eval()
    test_eval_out = task.test_eval()
    task.log('Best checkpoint (epoch={}, {}, {})'.format(
        task._passed_epoch, val_eval_out[BAR_KEY], test_eval_out[BAR_KEY]))

    task.log('Training time {}s'.format(time.time() - start_time))
