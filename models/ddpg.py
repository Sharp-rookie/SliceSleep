# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings;warnings.simplefilter('ignore')

from .common import Actor, Critic, ReplayBuffer


class DDPG(nn.Module):
    def __init__(self, lr, state_dim, action_dim, max_action, device='cpu'):
        super(DDPG, self).__init__()
        
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.buffer = ReplayBuffer()
        self.device = device
    
    def select_action(self, state):
        
        state = state.reshape(1, -1)
        with torch.no_grad():
            action = self.actor(state).flatten()
        return action
    
    def update(self, n_iter, batch_size, gamma, polyak):
        
        for i in range(n_iter):
            
            ####################
            # Sample a batch
            ####################
            state, action, reward, next_state, done = self.buffer.sample(batch_size)

            ####################
            # Actor Loss
            ####################
            actor_loss = -self.critic(state, self.actor(state)).mean()

            ####################
            # Critic Loss
            ####################
            # Select next action according to target actor with noise:
            next_action = (self.actor_target(next_state))

            # target Q-value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()

            # current Q-value
            current_Q = self.critic(state, action)
            loss_Q1 = F.mse_loss(current_Q, target_Q)

            ####################
            # Optimization
            ####################
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_optimizer.step()

            ####################
            # Polyak averaging update
            ####################
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                    
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, name))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
