# -*- coding=utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        self.size = (self.size+1) % self.max_size
        self.buffer.append(transition) # transiton is tuple of (state, action, reward, next_state, done)
    
    def sample(self, batch_size):
        
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
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)
        
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)
        
        self.max_action = max_action
        
    def forward(self, state):
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
        state_action = torch.cat([state, action], -1)
        
        q = F.relu(self.l1(state_action))
        q = self.ln1(q)
        q = F.relu(self.l2(q))
        q = self.ln2(q)
        q = self.l3(q)
        return q