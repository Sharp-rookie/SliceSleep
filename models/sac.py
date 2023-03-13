# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import warnings;warnings.simplefilter('ignore')

from .common import ReplayBuffer


################################## SAC Policy ##################################
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        out = self.net(state)
        return out
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(SoftQNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_actions, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        out = self.net(x)
        return out
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, max_action, log_std_min=-1, log_std_max=1):
        super(PolicyNetwork, self).__init__()
        
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.backbone = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.LayerNorm(64)
        )
        
        self.mean_linear = nn.Linear(64, num_actions)
        self.log_std_linear = nn.Linear(64, num_actions)
        
    def forward(self, state):

        x = self.backbone(state)
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        action = action * self.max_action
        
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        action  = action * self.max_action
        return action.flatten()


class SAC(nn.Module):
    def __init__(self, lr, state_dim, action_dim, max_action, device='cpu', log_dir=None, id=0):
        super(SAC, self).__init__()

        self.policy = PolicyNetwork(state_dim, action_dim, max_action, log_std_min=-1, log_std_max=1)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.value_net = ValueNetwork(state_dim)
        self.value_net_target = ValueNetwork(state_dim)
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=lr)

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=lr)

        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=lr)

        self.max_action = max_action
        self.buffer = ReplayBuffer(log_dir=log_dir, id=id)
        self.device = device

    def select_action(self, state):

        state = state.reshape(1, -1)
        with torch.no_grad():
            action = self.policy.get_action(state).flatten()
        return action

    def update(self, batch_size, gamma, polyak, mean_lambda=1e-3, std_lambda=1e-3, z_lambda=0.0):

        ####################
        # Sample a batch
        ####################
        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy.evaluate(state)

        ####################
        # Soft Q Loss
        ####################
        expected_q_value1 = self.soft_q_net1(state, action)
        expected_q_value2 = self.soft_q_net2(state, action)
        target_value = self.value_net_target(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss1 = F.mse_loss(expected_q_value1, next_q_value.detach())
        q_value_loss2 = F.mse_loss(expected_q_value2, next_q_value.detach())

        ####################
        # Value Loss
        ####################
        expected_min_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        next_value = expected_min_q_value - log_prob
        value_loss = F.mse_loss(expected_value, next_value.detach())

        ####################
        # Actor(Policy) Loss
        ####################
        expected_q_value = self.soft_q_net1(state, action)
        log_prob_target = expected_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        
        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss  = std_lambda  * log_std.pow(2).mean()
        z_loss    = z_lambda    * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        ####################
        # Optimization
        ####################
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
        ####################
        # Polyak averaging update
        ####################
        for target_param, param in zip(self.value_net_target.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - polyak) + param.data * polyak)
    
    def save(self, directory, name):
        torch.save(self.policy.state_dict(), '%s/%s_policy.pth' % (directory, name))
        
        torch.save(self.value_net.state_dict(), '%s/%s_value_net.pth' % (directory, name))
        torch.save(self.value_net_target.state_dict(), '%s/%s_value_net_target.pth' % (directory, name))
        
        torch.save(self.soft_q_net1.state_dict(), '%s/%s_soft_q_1.pth' % (directory, name))
        torch.save(self.soft_q_net2.state_dict(), '%s/%s_soft_q_2.pth' % (directory, name))
        
    def load(self, directory, name):
        self.policy.load_state_dict(torch.load('%s/%s_policy.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.value_net.load_state_dict(torch.load('%s/%s_value_net.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.value_net_target.load_state_dict(torch.load('%s/%s_value_net_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.soft_q_net1.load_state_dict(torch.load('%s/%s_soft_q_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.soft_q_net2.load_state_dict(torch.load('%s/%s_soft_q_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
                