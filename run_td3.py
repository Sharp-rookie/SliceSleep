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
from env import Environment


ue_n = 12 # UE数量，等效于负载等级，范围：3，6，9，12，15
log_dir = f"log/ue_{ue_n}/TD3/"
test_dir = f"test/ue_{ue_n}/TD3/"


def train():
    print("============================================================================================")

    ####### hyperparameters ######

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
    max_episodes = 10000        # max num of episodes
    max_timesteps = 150         # max timesteps in one episode

    #####################################################

    env = Environment(ue_number=[ue_n]*3)
    state_dim = state_n # state space dimension
    action_dim = action_space_n # action space dimension

    log_file = log_dir + "log" + ".csv"
    os.makedirs(log_dir, exist_ok=True)

    ################# training procedure ################

    # initialize a TD3 agent
    td3_agent1 = TD3(lr, state_dim, action_dim, max(action_space))
    td3_agent2 = TD3(lr, state_dim, action_dim, max(action_space))
    td3_agent3 = TD3(lr, state_dim, action_dim, max(action_space))

    # logging file
    log_f = open(log_file, "w+")
    log_f.write('episode,reward1,reward2,reward3\n')

    # tqdm
    bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
    pbar = tqdm(range(max_timesteps), desc='timesteps', bar_format=bar_format)
    
    # training loop
    start_time = datetime.now().replace(microsecond=0)
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
            td3_agent1.buffer.add((state, actions[0], rewards[0], next_state, float(dones[0])))
            td3_agent2.buffer.add((state, actions[1], rewards[1], next_state, float(dones[1])))
            td3_agent3.buffer.add((state, actions[2], rewards[2], next_state, float(dones[2])))
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
                td3_agent1.update(td3_agent1.buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                td3_agent2.update(td3_agent2.buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                td3_agent3.update(td3_agent3.buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                pbar.reset()
                break
            for i, done in enumerate(dones):
                if done:
                    if i == 0:
                        td3_agent1.update(td3_agent1.buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                    elif i == 1:
                        td3_agent2.update(td3_agent2.buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                    elif i == 2:
                        td3_agent3.update(td3_agent3.buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                    env.gnb.TD_policy.buckets[i].offset = 1
        
        # logging updates:
        log_f.write(f'{episode},{ep_reward[0]},{ep_reward[1]},{ep_reward[2]}\n')
        log_f.flush()
        
        # save the model
        if episode % save_interval == 0:
            ckpt_dir = log_dir + "checkpoint/episode" + str(episode) + "/"
            os.makedirs(ckpt_dir, exist_ok=True)
            td3_agent1.save(ckpt_dir, "slice1")
            td3_agent2.save(ckpt_dir, "slice2")
            td3_agent3.save(ckpt_dir, "slice3")
        
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

    env = Environment(ue_number=[ue_n]*3)
    state_dim = state_n
    action_dim = action_space_n
    max_action = max(action_space)


    ################# testing procedure ################

    # initialize TD3 agents
    td3_agent1 = TD3(0, state_dim, action_dim, max_action)
    td3_agent2 = TD3(0, state_dim, action_dim, max_action)
    td3_agent3 = TD3(0, state_dim, action_dim, max_action)
    td3_agent1.load_actor(path1)
    td3_agent2.load_actor(path2)
    td3_agent3.load_actor(path3)

    # track total testing time
    start_time = datetime.now().replace(microsecond=0)

    # logging file
    os.makedirs(test_dir, exist_ok=True)
    log_file = test_dir + "log.csv"
    log_f = open(log_file, "w+")
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
    plot_test_td3(log_file)
    return log_file


def plot_test_td3(path):
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

    plt.savefig(test_dir + 'result.jpg', dpi=300)


if __name__ == '__main__':

    train()
    test()