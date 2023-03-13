# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils import time_delta
from models import SAC


def train_sac(
        env,
        log_dir="log/ue_3/SAC/",
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

    ####### SAC hyperparameters ######

    state_dim = 6
    action_space = [-1, 0, 1]
    action_dim = len(action_space)

    polyak = 0.995      # target policy update parameter (1-tau)

    ################# training procedure ################

    # logging file
    log_file = log_dir + "log" + ".csv"
    os.makedirs(log_dir, exist_ok=True)
    log_f = open(log_file, "w+")
    log_f.write('episode,reward1,reward2,reward3\n')

    # initialize SAC agents
    sac_agent1 = SAC(lr, state_dim, action_dim, max(action_space), device=device, log_dir=log_dir, id=1).to(device)
    sac_agent2 = SAC(lr, state_dim, action_dim, max(action_space), device=device, log_dir=log_dir, id=2).to(device)
    sac_agent3 = SAC(lr, state_dim, action_dim, max(action_space), device=device, log_dir=log_dir, id=3).to(device)

    # tqdm
    bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
    pbar = tqdm(range(max_iters), desc='timesteps', bar_format=bar_format)

    # training loop
    start_time = datetime.now().replace(microsecond=0)
    for episode in range(1, max_episodes+1):

        ep_reward = [0 for _ in range(3)]

        env.reset()
        env.gnb.transP = 46
        state = torch.Tensor(env.get_state()).to(device)

        for iter in range(1, max_iters+1):

            # tqdm
            pbar.set_description(f"\33[36m🌌 Epoch {episode}/{max_episodes}")

            # select action and add exploration noise
            actions = [sac_agent1.select_action(state), sac_agent2.select_action(state), sac_agent3.select_action(state)]
            for i in range(3):
                actions[i] = actions[i] + torch.randn(action_dim).to(device)*exploration_noise
                actions[i] = actions[i].clip(min(action_space), max(action_space))
            
            # take action in env
            next_state, rewards, dones = env.step([action_space[torch.argmax(actions[0])], action_space[torch.argmax(actions[1])], action_space[torch.argmax(actions[2])]])
            next_state = torch.Tensor(next_state).to(device)
            rewards = torch.Tensor(rewards).to(device)
            dones = torch.Tensor(dones).float().to(device)

            # add to buffer
            sac_agent1.buffer.add((state, actions[0], rewards[0], next_state, dones[0]))
            sac_agent2.buffer.add((state, actions[1], rewards[1], next_state, dones[1]))
            sac_agent3.buffer.add((state, actions[2], rewards[2], next_state, dones[2]))
            state = next_state

            for i in range(3):
                ep_reward[i] += rewards[i]

            # tqdm
            cost_time = time_delta(start_time)
            pbar.set_postfix_str(f'{env.gnb.TD_policy.buckets[0].rate}tti, [{env.gnb.TD_policy.buckets[0].offset*100:.0f}%, {env.gnb.TD_policy.buckets[1].offset*100:.0f}%, {env.gnb.TD_policy.buckets[2].offset*100:.0f}%], [{round(env.gnb.datavolume[0]/1e3,2)}, {round(env.gnb.datavolume[1]/1e3,2)}, {round(env.gnb.datavolume[2]/1e3,2)}]B, [{env.gnb.delay[0]:.2f}, {env.gnb.delay[1]:.2f}, {env.gnb.delay[2]:.2f}]ms, [{rewards[0]:.2f}, {rewards[1]:.2f}, {rewards[2]:.2f}], {cost_time}\33[0m')
            pbar.update()

            # update sac_agent when episode is done
            if iter==(max_iters-1):
                sac_agent1.update(iter, batch_size, gamma, polyak)
                sac_agent2.update(iter, batch_size, gamma, polyak)
                sac_agent3.update(iter, batch_size, gamma, polyak)
                pbar.reset()
            for i, done in enumerate(dones):
                if done:
                    if i == 0:
                        sac_agent1.update(iter, batch_size, gamma, polyak)
                    elif i == 1:
                        sac_agent2.update(iter, batch_size, gamma, polyak)
                    elif i == 2:
                        sac_agent3.update(iter, batch_size, gamma, polyak)
                        
                    env.gnb.TD_policy.buckets[i].offset = 5*np.random.randint(1,20)/100
        
        # logging updates
        log_f.write(f'{episode},{ep_reward[0]},{ep_reward[1]},{ep_reward[2]}\n')
        log_f.flush()

        # save the model
        if episode % save_interval == 0:
            ckpt_dir = log_dir + "checkpoint/episode" + str(episode) + "/"
            os.makedirs(ckpt_dir, exist_ok=True)
            sac_agent1.save(ckpt_dir, "slice1")
            sac_agent2.save(ckpt_dir, "slice2")
            sac_agent3.save(ckpt_dir, "slice3")

        # noise decay
        if episode % noise_decay_freq == 0:
            exploration_noise  = max(exploration_noise-noise_decay_rate, min_noise)
            print('noise decay to: ', exploration_noise)

    log_f.close()
    env.close()


def test_sac(
        env,
        test_dir="test/ue_3/SAC/",
        max_episodes=1,
        max_iters=1500,
        device='cpu',
        path1 = f'checkpoints/SAC_15/2022_11_17_13_06_26/SAC_slice1_60_actor.pth',
        path2 = f'checkpoints/SAC_15/2022_12_01_12_36_09/SAC_slice2_60_actor.pth',
        path3 = f'checkpoints/SAC_15/2022_11_17_13_06_26/SAC_slice3_60_actor.pth'
    ):

    print("============================================================================================")

    ####### initialize environment hyperparameters ######

    state_dim = 6
    action_space = [-1, 0, 1]
    action_dim = len(action_space)
    max_action = max(action_space)

    ################# testing procedure ################

    # initialize SAC agents
    sac_agent1 = SAC(0, state_dim, action_dim, max_action)
    sac_agent2 = SAC(0, state_dim, action_dim, max_action)
    sac_agent3 = SAC(0, state_dim, action_dim, max_action)
    sac_agent1.load_actor(path1).to(device)
    sac_agent2.load_actor(path2).to(device)
    sac_agent3.load_actor(path3).to(device)

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
        env.reset()
        pbar.reset()

        env.gnb.transP = 46
        state = torch.Tensor(env.get_state()).to(device)

        for t in range(1, max_iters+1):

            # tqdm
            pbar.set_description(f"\33[36m🌌 Epoch {ep}/{max_episodes}")

            # select action with policy
            action = [sac_agent1.select_action(state), sac_agent2.select_action(state), sac_agent3.select_action(state)]
            next_state, rewards, _ = env.step([action_space[np.argmax(action[0])], action_space[np.argmax(action[1])], action_space[np.argmax(action[2])]])
            state = torch.Tensor([float(i) for i in next_state]).to(device)

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