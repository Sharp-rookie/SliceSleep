# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils import time_delta
from models import PPO
from env import Environment


ue_n = 12 # UE数量，等效于负载等级，范围：3，6，9，12，15
log_dir = f"log/ue_{ue_n}/PPO/"
test_dir = f"test/ue_{ue_n}/PPO/"


def train():

    ####### hyperparameters ######

    has_continuous_action_space = False        # continuous action space; else discrete
    state_n = 6
    action_space_n = 3
    action_space = [-1, 0, 1]
    max_ep_len = 150                           # max timesteps in one episode
    max_training_timesteps = max_ep_len * 1e3  # break training loop if timeteps > max_training_timesteps
    log_freq = max_ep_len                      # log avg reward in the interval (in num timesteps)
    save_model_freq = max_ep_len * 50          # save model frequency (in num timesteps)
    action_std = 0.6                           # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05               # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                       # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(5e3)           # action_std decay frequency (in num timesteps)
    update_timestep = max_ep_len      # update policy every n timesteps
    K_epochs = 80                     # update policy for K epochs in one PPO update
    eps_clip = 0.2                    # clip parameter for PPO
    gamma = 0.99                      # discount factor
    lr_actor = 0.0003                 # learning rate for actor network
    lr_critic = 0.001                 # learning rate for critic network

    #####################################################

    env = Environment(ue_number=[ue_n]*3)
    state_dim = state_n # state space dimension
    action_dim = action_space_n # action space dimension

    log_file = log_dir + "log" + ".csv"
    os.makedirs(log_dir, exist_ok=True)

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent1 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    ppo_agent2 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    ppo_agent3 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # retrain
    # path1 = f'checkpoints/PPO_9/2022_11_16_15_36_43/PPO_slice0_33.pth'
    # path2 = f'checkpoints/PPO_9/2022_11_16_04_40_38/PPO_slice1_29.pth'
    # path3 = f'checkpoints/PPO_9/2022_11_15_15_14_15/PPO_slice2_19.pth'
    # ppo_agent1.load(path1)
    # ppo_agent2.load(path2)
    # ppo_agent3.load(path3)
    # run_num_pretrained = 30

    # track total training time
    start_time = datetime.now().replace(microsecond=0)

    # logging file
    log_f = open(log_file, "w+")
    log_f.write('episode,timestep,reward1,datavolume1,reward2,datavolume2,reward3,datavolume3\n')

    # printing and logging variables
    log_running_reward1 = 0
    log_running_datavolume1 = 0
    log_running_reward2 = 0
    log_running_datavolume2 = 0
    log_running_reward3 = 0
    log_running_datavolume3 = 0
    log_running_episodes = 1
    time_step = 1
    i_episode = 1

    # tqdm
    bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
    pbar = tqdm(range(update_timestep), desc='timesteps', bar_format=bar_format)

    # training loop
    while time_step <= max_training_timesteps:

        state = [0] * state_dim
        env.reset()

        env.gnb.transP = 46

        for t in range(1, max_ep_len+1):

            # tqdm
            e = int(np.floor(time_step/update_timestep)) + 1
            pbar.set_description(f"\33[36m🌌 Epoch {e:2d}/{int(max_training_timesteps//update_timestep)}")

            # select action with policy
            action = [ppo_agent1.select_action(state), ppo_agent2.select_action(state), ppo_agent3.select_action(state)]
            state, rewards, dones = env.step([action_space[action[0]], action_space[action[1]], action_space[action[2]]])
            
            # saving reward and is_terminals
            ppo_agent1.buffer.rewards.append(rewards[0])
            ppo_agent2.buffer.rewards.append(rewards[1])
            ppo_agent3.buffer.rewards.append(rewards[2])
            ppo_agent1.buffer.is_terminals.append(dones[0])
            ppo_agent2.buffer.is_terminals.append(dones[1])
            ppo_agent3.buffer.is_terminals.append(dones[2])

            log_running_reward1 += rewards[0]
            log_running_datavolume1 += np.mean(env.datavolume[0])
            log_running_reward2 += rewards[1]
            log_running_datavolume2 += np.mean(env.datavolume[1])
            log_running_reward3 += rewards[2]
            log_running_datavolume3 += np.mean(env.datavolume[2])

            # tqdm
            cost_time = time_delta(start_time)
            pbar.set_postfix_str(f'{env.gnb.TD_policy.buckets[0].rate}tti, [{env.gnb.TD_policy.buckets[0].offset*100:.0f}%, {env.gnb.TD_policy.buckets[1].offset*100:.0f}%, {env.gnb.TD_policy.buckets[2].offset*100:.0f}%], [{round(env.datavolume[0],2)}, {round(env.datavolume[1],2)}, {round(env.datavolume[2],2)}]B, [{env.delay[0]:.2f}, {env.delay[1]:.2f}, {env.delay[2]:.2f}]ms, [{rewards[0]:.2f}, {rewards[1]:.2f}, {rewards[2]:.2f}], {cost_time}\33[0m')
            pbar.update()

            # update PPO agent
            time_step +=1
            if time_step % update_timestep == 0:
                ppo_agent1.update()
                ppo_agent2.update()
                ppo_agent3.update()
                pbar.reset()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent1.decay_action_std(action_std_decay_rate, min_action_std)
                ppo_agent2.decay_action_std(action_std_decay_rate, min_action_std)
                ppo_agent3.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward1 = log_running_reward1 / log_running_episodes
                log_avg_datavolume1 = log_running_datavolume1 / log_running_episodes
                log_avg_reward2 = log_running_reward2 / log_running_episodes
                log_avg_datavolume2 = log_running_datavolume2 / log_running_episodes
                log_avg_reward3 = log_running_reward3 / log_running_episodes
                log_avg_datavolume3 = log_running_datavolume3 / log_running_episodes

                log_f.write('{},{},{},{},{},{},{},{}\n'.format(i_episode, time_step, log_avg_reward1, log_avg_datavolume1, log_avg_reward2, log_avg_datavolume2, log_avg_reward3, log_avg_datavolume3))
                log_f.flush()

                log_running_reward1 = 0
                log_running_datavolume1 = 0
                log_running_reward2 = 0
                log_running_datavolume2 = 0
                log_running_reward3 = 0
                log_running_datavolume3 = 0
                log_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                ckpt_dir = log_dir + "checkpoint/time_step" + str(time_step) + "/"
                os.makedirs(ckpt_dir, exist_ok=True)
                ppo_agent1.save(ckpt_dir + 'slice1.pth')
                ppo_agent2.save(ckpt_dir + 'slice2.pth')
                ppo_agent3.save(ckpt_dir + 'slice3.pth')

            # break; if the episode is over
            for i, done in enumerate(dones):
                if done:
                    env.gnb.TD_policy.buckets[i].offset = 5*np.random.randint(1,20)/100

        log_running_episodes += 1
        i_episode += 1

    log_f.close()
    env.close()


def test():

    print('start ppo test')

    ####### initialize environment hyperparameters ######
    path1 = f'checkpoints/PPO_9/2022_11_16_15_36_43/PPO_slice0_33.pth'
    path2 = f'checkpoints/PPO_9/2022_11_17_00_45_53/PPO_slice1_43.pth'
    path3 = f'checkpoints/PPO_9/2022_11_15_15_14_15/PPO_slice2_19.pth'
    action_space_n = 3
    state_n = 6
    action_space = [-1, 0, 1]
    max_ep_len = 1500                           # max timesteps in one episode
    n_episodes = 1                             # break testing loop if timeteps > max_testing_timesteps

    #####################################################

    env = Environment(ue_number=[ue_n]*3)
    state_dim = state_n
    action_dim = action_space_n

    # initialize PPO agents
    ppo_agent1 = PPO(state_dim, action_dim, 0, 0, 0, 0, 0, 0, 0)
    ppo_agent2 = PPO(state_dim, action_dim, 0, 0, 0, 0, 0, 0, 0)
    ppo_agent3 = PPO(state_dim, action_dim, 0, 0, 0, 0, 0, 0, 0)
    ppo_agent1.load(path1)
    ppo_agent2.load(path2)
    ppo_agent3.load(path3)

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
            action = [ppo_agent1.select_action(state), ppo_agent2.select_action(state), ppo_agent3.select_action(state)]
            next_state, rewards, _ = env.step([action_space[action[0]], action_space[action[1]], action_space[action[2]]])
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
    plot_test_ppo(log_file)
    return log_file


def plot_test_ppo(path):
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