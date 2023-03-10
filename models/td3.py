# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        self.size += 1
        self.buffer.append(transition) # transiton is tuple of (state, action, reward, next_state, done)
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(s.unsqueeze(0))
            action.append(a.unsqueeze(0))
            reward.append(r.unsqueeze(0))
            next_state.append(s_.unsqueeze(0))
            done.append(d.unsqueeze(0))
        
        state = torch.concat(state,dim=0).reshape(batch_size,-1).detach()
        action = torch.concat(action,dim=0).reshape(batch_size,-1).detach()
        reward = torch.concat(reward,dim=0).reshape(batch_size,-1).detach()
        next_state = torch.concat(next_state,dim=0).reshape(batch_size,-1).detach()
        done = torch.concat(done,dim=0).reshape(batch_size,-1).detach()
                
        return state, action, reward, next_state, done


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, is_continuous):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)
        
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)
        
        self.max_action = max_action
        self.is_continuous = is_continuous
        
    def forward(self, state):
        if self.is_continuous:
            a = torch.tanh(self.l1(state))
            a = self.ln1(a)
            a = torch.tanh(self.l2(a))
            a = self.ln2(a)
            a = torch.sigmoid(self.l3(a)) * self.max_action
        else:
            a = F.relu(self.l1(state))
            a = self.ln1(a)
            a = F.relu(self.l2(a))
            a = self.ln2(a)
            a = torch.tanh(self.l3(a)) * self.max_action
            
        return a
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        
        q = F.relu(self.l1(state_action))
        q = self.ln1(q)
        q = F.relu(self.l2(q))
        q = self.ln2(q)
        q = self.l3(q)
        return q
    

class TD3(nn.Module):
    def __init__(self, lr, state_dim, action_dim, max_action, is_continuous=False, device='cpu'):
        super(TD3, self).__init__()
        
        self.actor = Actor(state_dim, action_dim, max_action, is_continuous)
        self.actor_target = Actor(state_dim, action_dim, max_action, is_continuous)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        self.max_action = max_action
        self.buffer = ReplayBuffer()
        self.device = device
    
    def select_action(self, state):
        state = state.reshape(1, -1)
        return self.actor(state).flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            
            # Select next action according to target policy:
            noise = action.data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = (reward + (1-done) * gamma * target_Q).detach()
            
            # Optimize Critic 1:
            self.critic_1_optimizer.zero_grad()
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            self.critic_2_optimizer.zero_grad()
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                    

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
    
    def load_actor(self, path1):
        self.actor.load_state_dict(torch.load(path1, map_location=lambda storage, loc: storage))
