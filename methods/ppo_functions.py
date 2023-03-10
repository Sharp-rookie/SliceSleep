# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils import time_delta
from models import PPO


def train_ppo(
        env,
        log_dir="log/ue_3/PPO/",
        save_interval=100,
        gamma = 0.99,
        batch_size=128,
        lr=0.01,
        max_episodes=10000,
        max_iters=150,
        exploration_noise=0.5,
        noise_decay_rate=0.05,
        min_noise=0.1,
        noise_decay_freq=50,
        device='cpu'
    ):

    print("============================================================================================")

    ####### PPO hyperparameters ######

    state_dim = 6
    action_space = [-1, 0, 1]
    action_dim = len(action_space)

    K_epochs = 80      # update policy for K epochs in one PPO update
    eps_clip = 0.2     # clip parameter for PPO
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    ################# training procedure ################

    # initialize PPO agents
    ppo_agent1 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device=device).to(device)
    ppo_agent2 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device=device).to(device)
    ppo_agent3 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device=device).to(device)

    # logging file
    log_file = log_dir + "log" + ".csv"
    os.makedirs(log_dir, exist_ok=True)
    log_f = open(log_file, "w+")
    log_f.write('episode,reward1,reward2,reward3\n')

    # tqdm
    bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
    pbar = tqdm(range(max_iters), desc='timesteps', bar_format=bar_format)

    # training loop
    start_time = datetime.now().replace(microsecond=0)
    for episode in range(1, max_episodes+1):

        state = torch.Tensor([0.0 for _ in range(state_dim)]).to(device)
        ep_reward = [0 for _ in range(3)]

        env.reset()
        env.gnb.transP = 46

        for iter in range(1, max_iters+1):

            # tqdm
            pbar.set_description(f"\33[36m🌌 Epoch {episode}/{max_episodes}")

            # select action with policy
            action = [ppo_agent1.select_action(state), ppo_agent2.select_action(state), ppo_agent3.select_action(state)]
            state, rewards, dones = env.step([action_space[action[0]], action_space[action[1]], action_space[action[2]]])
            state = torch.Tensor(state).to(device)
            rewards = torch.Tensor(rewards).to(device)
            dones = torch.Tensor(dones).float().to(device)
            
            # saving reward and is_terminals
            ppo_agent1.buffer.rewards.append(rewards[0])
            ppo_agent2.buffer.rewards.append(rewards[1])
            ppo_agent3.buffer.rewards.append(rewards[2])
            ppo_agent1.buffer.is_terminals.append(dones[0])
            ppo_agent2.buffer.is_terminals.append(dones[1])
            ppo_agent3.buffer.is_terminals.append(dones[2])

            for i in range(3):
                ep_reward[i] += rewards[i]

            # tqdm
            cost_time = time_delta(start_time)
            pbar.set_postfix_str(f'{env.gnb.TD_policy.buckets[0].rate}tti, [{env.gnb.TD_policy.buckets[0].offset*100:.0f}%, {env.gnb.TD_policy.buckets[1].offset*100:.0f}%, {env.gnb.TD_policy.buckets[2].offset*100:.0f}%], [{round(env.gnb.datavolume[0]/1e3,2)}, {round(env.gnb.datavolume[1]/1e3,2)}, {round(env.gnb.datavolume[2]/1e3,2)}]B, [{env.gnb.delay[0]:.2f}, {env.gnb.delay[1]:.2f}, {env.gnb.delay[2]:.2f}]ms, [{rewards[0]:.2f}, {rewards[1]:.2f}, {rewards[2]:.2f}], {cost_time}\33[0m')
            pbar.update()

            # update ppo_agent when episode is done
            if iter==(max_iters-1):
                ppo_agent1.update()
                ppo_agent2.update()
                ppo_agent3.update()
                pbar.reset()
            for i, done in enumerate(dones):
                if done:
                    if i == 0:
                        ppo_agent1.update()
                    elif i == 1:
                        ppo_agent2.update()
                    elif i == 2:
                        ppo_agent3.update()
                    env.gnb.TD_policy.buckets[i].offset = 1
        
        # logging updates:
        log_f.write(f'{episode},{ep_reward[0]},{ep_reward[1]},{ep_reward[2]}\n')
        log_f.flush()

        # save the model
        if episode % save_interval == 0:
            ckpt_dir = log_dir + "checkpoint/episode" + str(episode) + "/"
            os.makedirs(ckpt_dir, exist_ok=True)
            ppo_agent1.save(ckpt_dir + 'slice1.pth')
            ppo_agent2.save(ckpt_dir + 'slice2.pth')
            ppo_agent3.save(ckpt_dir + 'slice3.pth')

    log_f.close()
    env.close()


def test_ppo(
        env,
        test_dir="test/ue_3/PPO/",
        max_episodes=1,
        max_iters=1500,
        device='cpu',
        path1 = f'checkpoints/PPO_15/2022_11_17_13_06_26/PPO_slice1_60_actor.pth',
        path2 = f'checkpoints/PPO_15/2022_12_01_12_36_09/PPO_slice2_60_actor.pth',
        path3 = f'checkpoints/PPO_15/2022_11_17_13_06_26/PPO_slice3_60_actor.pth'
    ):

    print("============================================================================================")

    ####### initialize environment hyperparameters ######

    state_dim = 6
    action_space = [-1, 0, 1]
    action_dim = len(action_space)

    ################# testing procedure ################

    # initialize PPO agents
    ppo_agent1 = PPO(state_dim, action_dim)
    ppo_agent2 = PPO(state_dim, action_dim)
    ppo_agent3 = PPO(state_dim, action_dim)
    ppo_agent1.load(path1).to(device)
    ppo_agent2.load(path2).to(device)
    ppo_agent3.load(path3).to(device)

    # logging file
    os.makedirs(test_dir, exist_ok=True)
    log_file = test_dir + "log.csv"
    log_f = open(log_file, "w+")
    log_f.write('episode,timestep,offset1,datavolume1,delay1,offset2,datavolume2,delay2,offset3,datavolume3,delay3\n')

    # tqdm
    bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
    pbar = tqdm(range(max_iters), desc='timesteps', bar_format=bar_format)

    time_step = 0
    start_time = datetime.now().replace(microsecond=0)
    for ep in range(1, max_episodes+1):

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

        for t in range(1, max_iters+1):

            # tqdm
            pbar.set_description(f"\33[36m🌌 Epoch {ep}/{max_episodes}")

            # select action with policy
            action = [ppo_agent1.select_action(state), ppo_agent2.select_action(state), ppo_agent3.select_action(state)]
            next_state, rewards, _ = env.step([action_space[action[0]], action_space[action[1]], action_space[action[2]]])
            state = torch.Tensor(state).to(device)
            rewards = torch.Tensor(rewards).to(device)
            dones = torch.Tensor(dones).float().to(device)

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

        print('Episode: {}  Average Reward: [{}, {}, {}]'.format(ep, round(ep_reward[0]/max_iters, 2), round(ep_reward[1]/max_iters, 2), round(ep_reward[2]/max_iters, 2)))
        ep_reward = 0

    log_f.close()
    env.close()

    return log_file