import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd


LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
EPS = 1e-5
SCALE = 10
CLIP_EPS = 0.1

class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim

        self._build_actor()
        self._build_critic()

    def _build_actor(self):
        # pi network
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )
        self.actor_mean = nn.Linear(32, 1)
        self.actor_log_std = nn.Linear(32, 1)

    def _build_critic(self):
        # v network
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def act(self, state):
        act_hid = self.actor(state)

        mean = self.actor_mean(act_hid)
        log_std = self.actor_log_std(act_hid)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()

        normal = Normal(mean, std)

        action = normal.sample()
        logprob = normal.log_prob(action)

        return action.item(), logprob.item()

    def evaluate(self, state, action):
        # return the value of given state, and the probability of the actor take {action}
        state_value = self.critic(state)

        if action is None:
            return state_value, None

        act_hid = self.actor(state)

        mean = self.actor_mean(act_hid)
        log_std = self.actor_log_std(act_hid)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()

        normal = Normal(mean, std)

        action_logprob = normal.log_prob(action)

        return state_value, action_logprob


class PPO:
    def __init__(self, state_dim):
        self.policy = ActorCritic(state_dim).to(device)
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        # self.memory = Memory()

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        # make batch dim
        state = state.unsqueeze(dim=0)
        return self.policy.act(state)


class ActorDemo(nn.Module):
    def __init__(self, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size
        self.build_actor()

    def build_actor(self):
        self.actor = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )
        self.mu_mean = nn.Linear(32, 1)
        self.mu_log_std = nn.Linear(32, 1)
        self.sigma_mean = nn.Linear(32, 1)
        self.sigma_log_std = nn.Linear(32, 1)
    
    def action(self, state):
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

        mu = mu_normal.sample()
        sigma = sigma_normal.sample()
        logprob = mu_normal.log_prob(mu) * sigma_normal.log_prob(sigma)

        return mu.item(), sigma.item(), logprob.item()