import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import pandas as pd

from meta_sampler import MetaSamplerDataset


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

    def clear_mem(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.logprobs[:]


class MetaSampleEnv():
    def __init__(self, model, X, y, node_emb, edge_index, edge_weight):
        self.model = model
        self.X = X
        self.y = y
        self.node_emb = node_emb
        self.edge_index = edge_index
        self.edge_weight = edge_weight
    
    def reset(self):
        self.node_visit = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.meta_dataloader = self.make_sample_dataloader(self.X, self.y)
        self.meta_sampler = self.meta_dataloader.meta_sampler

    def make_sample_dataloader(self, X, y, shuffle=True):
        # return a data loader based on meta sampling
        dataset = MetaSamplerDataset(
            X, y, policy=None, num_nodes=self.node_emb.shape[0], 
            node_emb=self.node_emb, self.edge_index, self.edge_weight, 
            batch_size=32, subgraph_nodes=200, 
            shuffle=shuffle
        )

        return DataLoader(dataset, batch_size=None)
    
    def get_state(self):
        return self.meta_sampler.__get_init_state__()
    
    def step(self, state, action):
        return self.meta_sampler.step(state, action)
    
    def eval(self):
        loss = 1.0
        return loss
    

class ActorCritic(nn.Module):
    def __init__(self, state_size):
        super(ActorDemo, self).__init__()
        self.state_size = state_size
        self.build_actor()

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
        mu_mean, sigma_mean = action_mean
        sigma_mean, sigma_log_std = action_log_std

        mu_log_std = torch.clamp(mu_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        mu_std = mu_log_std.exp()
        sigma_log_std = torch.clamp(sigma_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        sigma_std = sigma_log_std.exp()

        mu_normal = Normal(mu_mean, mu_std)
        sigma_normal = Normal(sigma_mean, sigma_std)

        mu = mu_normal.sample()
        sigma = sigma_normal.sample()
        logprob = mu_normal.log_prob(mu) * sigma_normal.log_prob(sigma)

        action = [mu.item(), sigma.item()]

        return action, logprob.item()

    def _build_critic(self):
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

        mu_mean = self.mu_mean(act_hid)
        mu_log_std = self.mu_log_std(act_hid)
        sigma_mean = self.sigma_mean(act_hid)
        sigma_log_std = self.sigma_log_std(act_hid)

        mu_log_std = torch.clamp(mu_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        mu_std = mu_log_std.exp()
        sigma_log_std = torch.clamp(sigma_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        sigma_std = sigma_log_std.exp()

        mu_normal = Normal(mu_mean, mu_std)
        sigma_normal = Normal(sigma_mean, sigma_std)

        mu = action[0]
        sigma = action[1]
        action_logprob = mu_normal.log_prob(mu) * sigma_normal.log_prob(sigma)

        return state_value, action_logprob


class PPO:
    def __init__(self, state_size):
        self.policy = ActorCritic(state_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

        self.memory = Memory()

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        # make batch dim
        state = state.unsqueeze(dim=0)
        return self.policy.action(state)

    def make_batch(self):
        s = torch.FloatTensor(self.memory.states).to(device)
        a = torch.FloatTensor(self.memory.actions).to(device).unsqueeze(dim=1)
        r = torch.FloatTensor(self.memory.rewards).to(device).unsqueeze(dim=1)
        s_prime = torch.FloatTensor(self.memory.next_states).to(device)
        logp = torch.FloatTensor(self.memory.logprobs).to(device).\
            unsqueeze(dim=1)

        return s, a, r, s_prime, logp

    def train(self):
        s, a, r, s_prime, logp = self.make_batch()

        for i in range(nb_epoches):
            s_value, a_logprob = self.policy.evaluate(s, a)
            s_prime_value, _ = self.policy.evaluate(s_prime, None)

            td_target = r + gamma * s_prime_value
            delta = td_target - s_value
            delta = delta.detach().cpu().numpy()

            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lambda_ * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.FloatTensor(advantage_list).to(device)

            ratio = torch.exp(a_logprob - logp)

            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio,
                1 - CLIP_EPS,
                1 + CLIP_EPS
            ) * advantage

            loss = -torch.min(surr1, surr2) + \
                F.l1_loss(s_value, td_target.detach())

            print('Loss part 1: ', -torch.min(surr1, surr2).mean().item())
            print('Loss part 2: ', F.l1_loss(s_value, td_target.detach()).mean().item())
            print('Loss sum: ', loss.mean().item())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def train_step(self, env):
        for eid in range(nb_episodes):
            running_reward = 0
            env.reset()

            for i in range(nb_steps):
                a, logp = self.policy.select_action(s)
                s_prime, r = env.step(a)

                model.memory.states.append(s)
                model.memory.actions.append(a)
                model.memory.rewards.append(r)
                model.memory.next_states.append(s_prime)
                model.memory.logprobs.append(logp)

                s = s_prime
                running_reward += r

            model.train()
            model.memory.clear_mem()
            print('Reward in episode {}'.format(running_reward))
