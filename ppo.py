import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm

from meta_sampler import MetaSampler, MetaSamplerDataset


NUM_INIT_NODES = 10

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
EPS = 1e-5
SCALE = 10
CLIP_EPS = 0.1

nb_episodes = 3
nb_steps = 2
nb_epoches = 3

gamma = 0.98
lambda_ = 0.95


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.logprobs = []
        self.done_lst = []

    def clear_mem(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.done_lst[:]


class MetaSampleEnv():
    def __init__(self, model, X, y, num_nodes, node_emb, edge_index, edge_weight, subgraph_nodes, batch_size, device):
        self.model = model
        self.X = X
        self.y = y
        self.num_nodes = num_nodes
        self.node_emb = node_emb
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.subgraph_nodes = subgraph_nodes
        self.batch_size = batch_size
        self.device = device

        self.loss_func = nn.MSELoss()
    
    def reset(self):
        self.node_visit = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.meta_sampler = MetaSampler(None, self.num_nodes, self.node_emb, self.edge_index, self.subgraph_nodes, shuffle=True)
    
    def get_state(self):
        """get state vector of current subgraph"""
        cent_emb = self.node_emb[self.n_id].sum(dim=0)
        neighbor_emb = self.node_emb[self.neighbor_id].sum(dim=0)

        state = torch.cat([cent_emb, neighbor_emb], dim=0)
        return state.detach().cpu()
    
    def get_init_state(self):
        """Random init nodes and get neighbor, return state"""
        done = False
        unvisited_nodes_num = self.num_nodes - self.meta_sampler.node_visit.sum()
        if unvisited_nodes_num <= self.subgraph_nodes:
            # last subgraph
            self.n_id = np.where(self.meta_sampler.node_visit == False)[0]
            self.neighbor_id = self.n_id
            done = True
            
            subgraph = self.meta_sampler.__produce_subgraph_by_nodes__(self.n_id)
            self.meta_sampler.subgraphs.append(subgraph)
        else:
            self.n_id = self.meta_sampler.get_init_nodes()
            self.neighbor_id = self.meta_sampler.get_neighbor(self.n_id)
        
        s = self.get_state()

        return s, done
    
    def step(self, action):
        """extend current subgraph with an action"""
        sample_n_id = self.meta_sampler.neighbor_sample(self.neighbor_id, action)
        # avoid over subgraph_nodes(always intend to sample all neighbors)
        sample_n_id = self.meta_sampler.random_sample_left_nodes(self.n_id, sample_n_id)

        self.n_id = np.union1d(self.n_id, sample_n_id)
        done = False

        if len(self.n_id) >= self.subgraph_nodes:
            # last subgraph
            self.neighbor_id = self.n_id
            done = True

            subgraph = self.meta_sampler.__produce_subgraph_by_nodes__(self.n_id)
            self.meta_sampler.subgraphs.append(subgraph)
        else:
            self.neighbor_id = self.meta_sampler.get_neighbor(self.n_id)
        
        s_prime = self.get_state()

        return s_prime, done
    
    def finish(self):
        """subgraph nodes cover the whole graph"""
        if self.meta_sampler.node_visit.sum() == self.num_nodes:
            return True
        return False
    
    def make_dataloader(self):
        """return a data loader based on meta sampler"""
        dataset = MetaSamplerDataset(
            self.X, self.y, self.meta_sampler, self.num_nodes, 
            self.edge_index, self.edge_weight, 
            self.batch_size, shuffle=True, use_dist_sampler=False
        )

        return DataLoader(dataset, batch_size=None)

    def eval(self, episode):
        """evaluation and return loss as reward"""
        self.model.zero_grad()
        self.model.eval()

        dataloader = self.make_dataloader()
        pbar = tqdm(leave=False, dynamic_ncols=True)
        pbar.reset(total=len(dataloader))
        pbar.set_description(desc="Episode {}".format(episode))

        eval_outs = []
        for batch_idx, batch in enumerate(dataloader):
            batch = self.decorate_batch(batch)

            X, y, g, rows = batch
            with torch.no_grad():
                y_hat, _ = self.model(X, g)
            assert(y.size() == y_hat.size())

            batch_loss = self.loss_func(y, y_hat)
            eval_outs.append(batch_loss)
            self.process_bar(pbar, batch_loss.item(), inc=True)
            

        eval_loss = torch.stack(eval_outs).mean()
        self.process_bar(pbar, eval_loss.item(), inc=False)
        pbar.close()
        self.model.train()

        return eval_loss
    
    def process_bar(self, pbar, loss, inc=True):
        bar_info = {'val_loss': loss}
        pbar.set_postfix(**bar_info)

        if inc:
            pbar.update(n=1)
    
    def decorate_batch(self, batch):
        # Decorate the input batch with a proper device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
            return batch
        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
                elif isinstance(value, dict) or isinstance(value, list):
                    batch[key] = self.decorate_batch(value)
                # retain other value types in the batch dict
            return batch
        elif isinstance(batch, list):
            new_batch = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    new_batch.append(value.to(self.device))
                elif isinstance(value, dict) or isinstance(value, list):
                    new_batch.append(self.decorate_batch(value))
                else:
                    # retain other value types in the batch list
                    new_batch.append(value)
            return new_batch
        else:
            raise Exception('Unsupported batch type {}'.format(type(batch)))
    

class ActorCritic(nn.Module):
    def __init__(self, state_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.build_actor()
        self.build_critic()

    def build_actor(self):
        self.actor = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU()
        )
        self.action_mean = nn.Linear(32, 2)
        self.action_log_std = nn.Linear(32, 2)
    
    def action(self, state):
        act_hid = self.actor(state)

        action_mean = self.action_mean(act_hid)
        action_log_std = self.action_log_std(act_hid)
        mu_mean, sigma_mean = action_mean[0], action_mean[1]
        mu_log_std, sigma_log_std = action_log_std[0], action_log_std[1]

        mu_log_std = torch.clamp(mu_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        mu_std = mu_log_std.exp()
        sigma_log_std = torch.clamp(sigma_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        sigma_std = sigma_log_std.exp()

        mu_normal = Normal(mu_mean, mu_std)
        sigma_normal = Normal(sigma_mean, sigma_std)

        mu = mu_normal.sample()
        sigma = sigma_normal.sample()
        logprob = mu_normal.log_prob(mu) + sigma_normal.log_prob(sigma)

        action = [mu.item(), sigma.item()]

        return action, logprob.item()

    def build_critic(self):
        # v network
        self.critic = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def evaluate(self, state, action):
        # return the value of given state, and the probability of the actor take {action}
        state_value = self.critic(state)

        if action is None:
            return state_value, None

        act_hid = self.actor(state)

        action_mean = self.action_mean(act_hid)
        action_log_std = self.action_log_std(act_hid)
        mu_mean, sigma_mean = action_mean[0], action_mean[1]
        mu_log_std, sigma_log_std = action_log_std[0], action_log_std[1]

        mu_log_std = torch.clamp(mu_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        mu_std = mu_log_std.exp()
        sigma_log_std = torch.clamp(sigma_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        sigma_std = sigma_log_std.exp()

        mu_normal = Normal(mu_mean, mu_std)
        sigma_normal = Normal(sigma_mean, sigma_std)

        mu, sigma = action[0], action[1]
        # log(pq) = log(p) + log(q)
        action_logprob = mu_normal.log_prob(mu) + sigma_normal.log_prob(sigma)

        return state_value, action_logprob


class PPO:
    def __init__(self, state_size, device):
        self.policy = ActorCritic(state_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

        self.memory = Memory()
        self.device = device

    def make_batch(self):
        # Data format
        # state is a list of tensor
        # action is a list of list
        # others are a list of single value
        s = torch.stack(self.memory.states).to(self.device)
        a = torch.FloatTensor(self.memory.actions).to(self.device)
        r = torch.FloatTensor(self.memory.rewards).to(self.device).unsqueeze(dim=1)
        s_prime = torch.stack(self.memory.next_states).to(self.device)
        logp = torch.FloatTensor(self.memory.logprobs).to(self.device).unsqueeze(dim=1)
        done_mask = torch.FloatTensor(self.memory.done_lst).to(self.device).unsqueeze(dim=1)

        return s, a, r, s_prime, logp, done_mask

    def train(self):
        s, a, r, s_prime, logp, done_mask = self.make_batch()
        self.policy.to(self.device)

        for i in range(nb_epoches):
            s_value, a_logprob = self.policy.evaluate(s, a)
            s_prime_value, _ = self.policy.evaluate(s_prime, None)

            td_target = r + gamma * s_prime_value * done_mask
            delta = td_target - s_value
            delta = delta.detach().cpu().numpy()

            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lambda_ * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.FloatTensor(advantage_list).to(self.device)

            ratio = torch.exp(a_logprob - logp)

            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio,
                1 - CLIP_EPS,
                1 + CLIP_EPS
            ) * advantage

            loss = -torch.min(surr1, surr2) + \
                F.l1_loss(s_value, td_target.detach())

            # print('Loss part 1: ', -torch.min(surr1, surr2).mean().item())
            # print('Loss part 2: ', F.l1_loss(s_value, td_target.detach()).mean().item())
            # print('Loss sum: ', loss.mean().item())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.policy.cpu()

    def train_step(self, env):
        for eid in range(nb_episodes):
            env.reset()
            action_cnt = 0

            # get all subgraphs
            while not env.finish():
                done = False
                s, done = env.get_init_state()
                if done:
                    break

                # get a subgraph step by step
                for i in range(nb_steps):
                    a, logp = self.policy.action(s)
                    s_prime, done = env.step(a)

                    self.memory.states.append(s)
                    self.memory.actions.append(a)
                    self.memory.next_states.append(s_prime)
                    self.memory.logprobs.append(logp)
                    done_mask = 0 if done else 1
                    self.memory.done_lst.append(done_mask)

                    s = s_prime
                    action_cnt += 1

                    if done:
                        break

            r = -env.eval(eid+1)
            for i in range(action_cnt):
                self.memory.rewards.append(r)
            self.train()
            self.memory.clear_mem()
