# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils import time_delta
from models import TD3
from env import Environment, current_t


class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        self.size +=1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)


def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######

    env_name = "td3_env"
    state_n = 6
    action_space_n = 3
    action_space = [-1, 0, 1]
    save_interval = 100
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.003
    exploration_noise = 0.4
    noise_decay_rate = 0.04
    min_exploration_noise = 0.1
    exploration_noise_decay_freq = 50
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 10000         # max num of episodes
    max_timesteps = 150          # max timesteps in one episode

    #####################################################

    ue_number = 12
    env = Environment(ue_number=[ue_number]*3)
    # state space dimension
    state_dim = state_n
    # action space dimension
    action_dim = action_space_n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = f"log/TD3_logs_{ue_number}/"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    #### create new log file for each run
    log_f_name = log_dir + "log_" + str(run_num) + ".csv"
    print("logging at : " + log_f_name)
    
    ################### checkpointing ###################

    run_num_pretrained = 0
    directory = f"checkpoints/TD3_{ue_number}/{current_t}/"
    if not os.path.exists(directory):
          os.makedirs(directory)
    checkpoint_path = "TD3_{}".format(run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    
    ################# training procedure ################

    # initialize a TD3 agent
    td3_agent1 = TD3(lr, state_dim, action_dim, max(action_space))
    td3_agent2 = TD3(lr, state_dim, action_dim, max(action_space))
    td3_agent3 = TD3(lr, state_dim, action_dim, max(action_space))

    # retrain
    # td3_agent1.load('checkpoints/TD3_12/2022_11_17_13_06_22', 'TD3_slice1_81')
    # td3_agent2.load('checkpoints/TD3_12/2022_11_17_13_06_22', 'TD3_slice2_81')
    # td3_agent3.load('checkpoints/TD3_12/2022_11_17_13_06_22', 'TD3_slice3_81')
    # run_num_pretrained = 81

    # memory pool
    replay_buffer1 = ReplayBuffer()
    replay_buffer2 = ReplayBuffer()
    replay_buffer3 = ReplayBuffer()
    
    # track total training time
    start_time = datetime.now().replace(microsecond=0)

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,reward1,reward2,reward3\n')

    # tqdm
    bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
    pbar = tqdm(range(max_timesteps), desc='timesteps', bar_format=bar_format)
    
    # training loop
    for episode in range(1, max_episodes+1):
        
        state = torch.tensor([0.0] * state_dim)
        ep_reward = [0]*3
        ep_datavolume = [0]*3
        env.reset()

        env.gnb.transP = 46
        
        for t in range(max_timesteps):
            
            # tqdm
            pbar.set_description(f"\33[36m🌌 Epoch {episode}/{max_episodes}")
            
            # select action and add exploration noise:
            actions = [td3_agent1.select_action(state), td3_agent2.select_action(state), td3_agent3.select_action(state)]
            for i in range(3):
                actions[i] = actions[i] + np.random.normal(0, exploration_noise, size=action_dim)
                actions[i] = actions[i].clip(min(action_space), max(action_space))
            
            # take action in env:
            next_state, rewards, dones = env.step([action_space[np.argmax(actions[0])], action_space[np.argmax(actions[1])], action_space[np.argmax(actions[2])]])
            next_state = torch.tensor([float(i) for i in next_state])
            replay_buffer1.add((state, actions[0], rewards[0], next_state, float(dones[0])))
            replay_buffer2.add((state, actions[1], rewards[1], next_state, float(dones[1])))
            replay_buffer3.add((state, actions[2], rewards[2], next_state, float(dones[2])))
            state = next_state
            
            for i in range(3):
                ep_reward[i] += rewards[i]
                ep_datavolume[i] += np.mean(env.datavolume[i])

            # tqdm
            cost_time = time_delta(start_time)
            pbar.set_postfix_str(f'{env.gnb.TD_policy.buckets[0].rate}tti, [{env.gnb.TD_policy.buckets[0].offset*100:.0f}%, {env.gnb.TD_policy.buckets[1].offset*100:.0f}%, {env.gnb.TD_policy.buckets[2].offset*100:.0f}%], [{round(env.gnb.datavolume[0]/1e3,2)}, {round(env.gnb.datavolume[1]/1e3,2)}, {round(env.gnb.datavolume[2]/1e3,2)}]B, [{env.gnb.delay[0]:.2f}, {env.gnb.delay[1]:.2f}, {env.gnb.delay[2]:.2f}]ms, [{rewards[0]:.2f}, {rewards[1]:.2f}, {rewards[2]:.2f}], {cost_time}\33[0m')
            pbar.update()
            
            # if episode is done then update td3_agent:
            if t==(max_timesteps-1):
                td3_agent1.update(replay_buffer1, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                td3_agent2.update(replay_buffer2, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                td3_agent3.update(replay_buffer3, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                pbar.reset()
                break
            for i, done in enumerate(dones):
                if done:
                    if i == 0:
                        td3_agent1.update(replay_buffer1, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                    elif i == 1:
                        td3_agent2.update(replay_buffer2, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                    elif i == 2:
                        td3_agent3.update(replay_buffer3, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                    env.gnb.TD_policy.buckets[i].offset = 1
        
        # logging updates:
        log_f.write(f'{episode},{ep_reward[0]},{ep_reward[1]},{ep_reward[2]}\n')
        log_f.flush()
        
        # save the model
        if episode % save_interval == 0:
            run_num_pretrained += 1
            td3_agent1.save(directory, "TD3_slice{}_{}".format(1, run_num_pretrained))
            td3_agent2.save(directory, "TD3_slice{}_{}".format(2, run_num_pretrained))
            td3_agent3.save(directory, "TD3_slice{}_{}".format(3, run_num_pretrained))
        
        # noise decay
        if episode % exploration_noise_decay_freq == 0:
            exploration_noise  = max(exploration_noise-noise_decay_rate, min_exploration_noise)
            print('noise decay to: ', exploration_noise)

    log_f.close()
    env.close()


def test():

    print('start td3 test')

    ####### initialize environment hyperparameters ######
    path1 = f'checkpoints/TD3_15/2022_11_17_13_06_26/TD3_slice1_60_actor.pth'
    path2 = f'checkpoints/TD3_15/2022_12_01_12_36_09/TD3_slice2_5_actor.pth'
    path3 = f'checkpoints/TD3_15/2022_11_17_13_06_26/TD3_slice3_60_actor.pth'
    action_space_n = 3
    state_n = 6
    action_space = [-1, 0, 1]
    max_ep_len = 1500                           # max timesteps in one episode
    n_episodes = 1                             # break training loop if timeteps > max_training_timesteps
    
    #####################################################

    ue_number = 15
    env = Environment(ue_number=[ue_number]*3)
    state_dim = state_n
    action_dim = action_space_n
    max_action = max(action_space)

    #### log files for multiple runs are NOT overwritten
    log_dir = f"test/TD3_logs_{ue_number}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    #### create new log file for each run
    log_f_name = log_dir + "log_" + str(run_num) + ".csv"
    print("logging at : " + log_f_name)

    ################# training procedure ################

    # initialize TD3 agents
    td3_agent1 = TD3(0, state_dim, action_dim, max_action)
    td3_agent2 = TD3(0, state_dim, action_dim, max_action)
    td3_agent3 = TD3(0, state_dim, action_dim, max_action)
    td3_agent1.load_actor(path1)
    td3_agent2.load_actor(path2)
    td3_agent3.load_actor(path3)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,offset1,datavolume1,delay1,offset2,datavolume2,delay2,offset3,datavolume3,delay3\n')

    # tqdm
    bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
    pbar = tqdm(range(max_ep_len), desc='timesteps', bar_format=bar_format)

    time_step = 0
    for ep in range(1, n_episodes+1):

        # statistic
        datavolume = [0]*3
        delay = [0]*3
        offset = [0]*3
        ep_reward = [0]*3

        # reset
        state = torch.tensor([0.0] * state_dim)
        env.reset()
        pbar.reset()

        env.gnb.transP = 46

        for t in range(1, max_ep_len+1):

            # tqdm
            pbar.set_description(f"\33[36m🌌 Epoch {ep}/{n_episodes}")

            # select action with policy
            action = [td3_agent1.select_action(state), td3_agent2.select_action(state), td3_agent3.select_action(state)]
            next_state, rewards, _ = env.step([action_space[np.argmax(action[0])], action_space[np.argmax(action[1])], action_space[np.argmax(action[2])]])
            state = torch.tensor([float(i) for i in next_state])

            # statistic
            for i in range(3):
                ep_reward[i] += rewards[i]
                datavolume[i] = np.mean(env.datavolume[i])
                delay[i] = np.mean(env.delay[i])
                offset[i] = env.gnb.TD_policy.buckets[i].offset

            # tqdm
            cost_time = time_delta(start_time)
            pbar.set_postfix_str(f'{env.gnb.TD_policy.buckets[0].rate}tti, [{env.gnb.TD_policy.buckets[0].offset*100:.0f}%, {env.gnb.TD_policy.buckets[1].offset*100:.0f}%, {env.gnb.TD_policy.buckets[2].offset*100:.0f}%], [{round(env.gnb.datavolume[0]/1e3,2)}, {round(env.gnb.datavolume[1]/1e3,2)}, {round(env.gnb.datavolume[2]/1e3,2)}]kB, [{env.gnb.delay[0]:.2f}, {env.gnb.delay[1]:.2f}, {env.gnb.delay[2]:.2f}]ms, [{rewards[0]:.2f}, {rewards[1]:.2f}, {rewards[2]:.2f}], {cost_time}\33[0m')
            pbar.update()

            # log in logging file
            time_step +=1
            log_f.write(f'{ep},{time_step},{offset[0]},{datavolume[0]},{delay[0]},{offset[1]},{datavolume[1]},{delay[1]},{offset[2]},{datavolume[2]},{delay[2]}\n')
            log_f.flush()

        print('Episode: {}  Average Reward: [{}, {}, {}]'.format(ep, round(ep_reward[0]/max_ep_len, 2), round(ep_reward[1]/max_ep_len, 2), round(ep_reward[2]/max_ep_len, 2)))
        ep_reward = 0

    log_f.close()
    env.close()
    plot_test_td3(log_f_name, ue_number)
    return log_f_name

def plot_test_td3(path, ue_number):
    log = pd.read_csv(path)
    plt.figure(figsize=(36, 12))
    plt.title('Test')

    t = log['timestep']
    datavolume = [log['datavolume1'], log['datavolume2'], log['datavolume3']]
    offset = [log['offset1'], log['offset2'], log['offset3']]
    delay = [log['delay1'], log['delay2'], log['delay3']]

    qos = [30, 300, 100]
    for i in range(3):

        qos_count = 0
        qos_ratio = []

        plt.subplot(3,3,3*i+1)
        plt.title(f'offset{i+1}')
        plt.plot(t, offset[i], label='offset')
        plt.legend()

        plt.subplot(3,3,3*i+2)
        plt.title(f'delay{i+1}')
        plt.plot(t, delay[i], label='delay')
        plt.plot(t, [qos[i]]*len(t), label='qos')
        plt.legend()

        plt.subplot(3,3,3*i+3)
        plt.title(f'qos{i+1}')
        for j in range(len(delay[i])):
            qos_count += delay[i][j] < qos[i]
            qos_ratio.append(qos_count/(j+1))
        plt.plot(t, qos_ratio, label='qos satification')
        plt.legend()

    plt.savefig(f'test/td3_ue{ue_number}.jpg', dpi=300)
    print('save line graph at test/td3.jpg')


if __name__ == '__main__':

    train()
    test()