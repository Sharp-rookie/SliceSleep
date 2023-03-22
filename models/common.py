# -*- coding=utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_normal(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0., std=0.01)
        nn.init.constant_(layer.bias, 0.)


class ReplayBuffer:
    def __init__(self, max_size=5e5, log_dir=None, id=0):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

        if log_dir:
            log_file = log_dir + f"pool{id}" + ".csv"
            self.log_f = open(log_file, "w+")
            self.log_f.write('dv1,dv2,dv3,offset1,offset2,offset3,act-1,act+0,act+1,reward,n_dv1,n_dv2,n_dv3,n_offset1,n_offset2,n_offset3,done,\n')
        else:
            self.log_f = None
    
    def add(self, transition):
        self.size = (self.size+1) % self.max_size
        self.buffer.append(transition) # transiton is tuple of (state, action, reward, next_state, done)

        if self.log_f:
            for item in transition:
                if isinstance(item.cpu().tolist(), list):
                    for i in item:
                        self.log_f.write(f'{i},')
                elif isinstance(item.cpu().tolist(), float):
                    self.log_f.write(f'{item},')
            self.log_f.write(f'\n')   
            self.log_f.flush()
    
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
        
        state = torch.concat(state,dim=0).reshape(batch_size,-1)
        action = torch.concat(action,dim=0).reshape(batch_size,-1)
        reward = torch.concat(reward,dim=0).reshape(batch_size,-1)
        next_state = torch.concat(next_state,dim=0).reshape(batch_size,-1)
        done = torch.concat(done,dim=0).reshape(batch_size,-1)
                
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