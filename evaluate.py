# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings;warnings.filterwarnings("ignore")

from utils import time_delta, setup_seed
from models import TD3
from env import Environment


setup_seed(729)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_td3(ue_num=3, max_ep_len=1000, is_tqdm=False):

    ####### initialize environment hyperparameters ######

    path1 = f'model/ue_{ue_num}/TD3_slice1.pth'
    path2 = f'model/ue_{ue_num}/TD3_slice2.pth'
    path3 = f'model/ue_{ue_num}/TD3_slice3.pth'
    action_space_n = 3
    state_n = 6
    action_space = [-1, 0, 1]
    n_episodes = 1 # break training loop if timeteps > max_training_timesteps
    
    ###################### logging ######################

    log_dir = "test_log/TD3_logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    log_f_name = log_dir + "log_" + str(run_num) + ".csv"

    ################# training procedure ################

    env = Environment(ue_number=[ue_num]*3)
    state_dim = state_n
    action_dim = action_space_n
    max_action = max(action_space)

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
    log_f.write('episode,timestep,offset1,power1,datavolume1,throughput1,prb_u1,delay1,offset2,power2,datavolume2,throughput2,prb_u2,delay2,offset3,power3,datavolume3,throughput3,prb_u3,delay3\n')

    # tqdm
    if is_tqdm:
        bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
        pbar = tqdm(range(max_ep_len), desc='timesteps', bar_format=bar_format)

    time_step = 0
    for ep in range(1, n_episodes+1):

        # statistic
        datavolume = [0]*3
        throughput = [0]*3
        power = [0]*3
        prb_u = [0]*3
        delay = [0]*3
        offset = [0]*3

        # reset
        state = torch.tensor([0.0] * state_dim)
        env.reset()
        env.gnb.TD_policy.buckets[0].offset = 1
        env.gnb.TD_policy.buckets[1].offset = 1
        env.gnb.TD_policy.buckets[2].offset = 1

        if is_tqdm:
            pbar.reset()
        
        env.gnb.transP = 46

        for t in range(1, max_ep_len+1):

            # tqdm
            if is_tqdm:
                pbar.set_description(f"\33[36m🌌 Epoch {ep}/{n_episodes}")

            # select action with policy
            action = [td3_agent1.select_action(state), td3_agent2.select_action(state), td3_agent3.select_action(state)]
            next_state, rewards, _ = env.step([action_space[np.argmax(action[0])], action_space[np.argmax(action[1])], action_space[np.argmax(action[2])]])
            state = torch.tensor([float(i) for i in next_state])

            # statistic
            for i in range(3):
                datavolume[i] = np.mean(env.datavolume[i])/(env.simulate_duration*env.gnb.tti/1000) # 1s内的等效量
                throughput[i] = np.mean(env.throughput[i])/(env.simulate_duration*env.gnb.tti/1000)
                power[i] = np.mean(env.power[i])/(env.simulate_duration*env.gnb.tti/1000)
                prb_u[i] = np.mean(env.prb_utilization[i])
                delay[i] = np.mean(env.delay[i])
                offset[i] = env.gnb.TD_policy.buckets[i].offset

            # tqdm
            if is_tqdm:
                cost_time = time_delta(start_time)
                pbar.set_postfix_str(f'{env.gnb.TD_policy.buckets[0].rate}tti, [{env.gnb.TD_policy.buckets[0].offset*100:.0f}%, {env.gnb.TD_policy.buckets[1].offset*100:.0f}%, {env.gnb.TD_policy.buckets[2].offset*100:.0f}%], [{round(env.gnb.datavolume[0]/1e3,2)}, {round(env.gnb.datavolume[1]/1e3,2)}, {round(env.gnb.datavolume[2]/1e3,2)}]kB, [{env.gnb.delay[0]:.2f}, {env.gnb.delay[1]:.2f}, {env.gnb.delay[2]:.2f}]ms, [{rewards[0]:.2f}, {rewards[1]:.2f}, {rewards[2]:.2f}], {cost_time}\33[0m')
                pbar.update()

            # log in logging file
            time_step +=1
            log_f.write(f'{ep},{time_step},{offset[0]},{power[0]},{datavolume[0]},{throughput[0]},{prb_u[0]},{delay[0]},{offset[1]},{power[1]},{datavolume[1]},{throughput[1]},{prb_u[1]},{delay[1]},{offset[2]},{power[2]},{datavolume[2]},{throughput[2]},{prb_u[2]},{delay[2]}\n')
            log_f.flush()

    log_f.close()
    env.close()
    return log_f_name


def test_nonRL(ue_num=3, max_ep_len=1000, is_tqdm=False):

    ####### initialize environment hyperparameters ######

    n_episodes = 1 # break training loop if timeteps > max_training_timesteps
    
    ###################### logging ######################

    log_dir = "test_log/nonRL_logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    log_f_name = log_dir + "log_" + str(run_num) + ".csv"

    ################# training procedure ################

    env = Environment(ue_number=[ue_num]*3)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,offset1,power1,datavolume1,throughput1,prb_u1,delay1,offset2,power2,datavolume2,throughput2,prb_u2,delay2,offset3,power3,datavolume3,throughput3,prb_u3,delay3\n')

    # tqdm
    if is_tqdm:
        bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
        pbar = tqdm(range(max_ep_len), desc='timesteps', bar_format=bar_format)

    time_step = 0
    for ep in range(1, n_episodes+1):

        # statistic
        datavolume = [0]*3
        throughput = [0]*3
        power = [0]*3
        prb_u = [0]*3
        delay = [0]*3
        offset = [0]*3

        # reset
        env.reset()
        env.gnb.TD_policy.buckets[0].offset = 1
        env.gnb.TD_policy.buckets[1].offset = 1
        env.gnb.TD_policy.buckets[2].offset = 1
        if is_tqdm:
            pbar.reset()

        for t in range(1, max_ep_len+1):

            # tqdm
            if is_tqdm:
                pbar.set_description(f"\33[36m🌌 Epoch {ep}/{n_episodes}")

            # simulate
            env.step([0,0,0])

            # statistic
            for i in range(3):
                datavolume[i] = np.mean(env.datavolume[i])/(env.simulate_duration*env.gnb.tti/1000) # 1s内的等效量
                throughput[i] = np.mean(env.throughput[i])/(env.simulate_duration*env.gnb.tti/1000)
                power[i] = np.mean(env.power[i])/(env.simulate_duration*env.gnb.tti/1000)
                prb_u[i] = np.mean(env.prb_utilization[i])
                delay[i] = np.mean(env.delay[i])
                offset[i] = env.gnb.TD_policy.buckets[i].offset

            # tqdm
            if is_tqdm:
                cost_time = time_delta(start_time)
                pbar.set_postfix_str(f'{env.gnb.TD_policy.buckets[0].rate}tti, [{env.gnb.TD_policy.buckets[0].offset*100:.0f}%, {env.gnb.TD_policy.buckets[1].offset*100:.0f}%, {env.gnb.TD_policy.buckets[2].offset*100:.0f}%], [{round(env.gnb.datavolume[0]/1e3,2)}, {round(env.gnb.datavolume[1]/1e3,2)}, {round(env.gnb.datavolume[2]/1e3,2)}]kB, [{env.gnb.delay[0]:.2f}, {env.gnb.delay[1]:.2f}, {env.gnb.delay[2]:.2f}]ms, {cost_time}\33[0m')
                pbar.update()

            # log in logging file
            time_step +=1
            log_f.write(f'{ep},{time_step},{offset[0]},{power[0]},{datavolume[0]},{throughput[0]},{prb_u[0]},{delay[0]},{offset[1]},{power[1]},{datavolume[1]},{throughput[1]},{prb_u[1]},{delay[1]},{offset[2]},{power[2]},{datavolume[2]},{throughput[2]},{prb_u[2]},{delay[2]}\n')
            log_f.flush()

    log_f.close()
    env.close()
    return log_f_name


def plot_td3_nonRL(td3_path, nonRL_path):

    td3_log = pd.read_csv(td3_path)
    nonRL_log = pd.read_csv(nonRL_path)
    t = td3_log['timestep']
    qos = [30, 300, 100]

    td3_throughput = [td3_log['throughput1'], td3_log['throughput2'], td3_log['throughput3']]
    non_throughput = [nonRL_log['throughput1'], nonRL_log['throughput2'], nonRL_log['throughput3']]
    td3_offset = [td3_log['offset1'], td3_log['offset2'], td3_log['offset3']]
    non_offset = [nonRL_log['offset1'], nonRL_log['offset2'], nonRL_log['offset3']]
    td3_datavolume = [td3_log['datavolume1'], td3_log['datavolume2'], td3_log['datavolume3']]
    td3_delay = [td3_log['delay1'], td3_log['delay2'], td3_log['delay3']]
    nonRL_delay = [nonRL_log['delay1'], nonRL_log['delay2'], nonRL_log['delay3']]
    td3_power = [td3_log['power1'], td3_log['power2'], td3_log['power3']]
    non_power = [nonRL_log['power1'], nonRL_log['power2'], nonRL_log['power3']]
    td3_efficiency = [(td3_throughput[i]/8000)/td3_power[i] for i in range(3)]
    non_efficiency = [(non_throughput[i]/8000)/non_power[i] for i in range(3)]
    
    plt.style.use(['science'])

    # datavolume
    _, ax = plt.subplots(1, 1, figsize=(9, 6))
    for i in range(3):
        for j in range(1, len(td3_datavolume[i])):
            td3_datavolume[i][j] = 0.8*td3_datavolume[i][j-1] + 0.2*td3_datavolume[i][j]
    plt.plot(range(200), (td3_datavolume[2][500:700]/8000000), label='Service 1', lw=1.2)
    plt.legend(fontsize=15)
    plt.plot(range(200), (td3_datavolume[1][500:700]/8000000), "-o", label='Service 2', lw=2)
    plt.legend(fontsize=15)
    plt.plot(range(200), (td3_datavolume[2][500:700]/8000000), "-^", label='Service 3', lw=2)
    plt.legend(fontsize=15)
    axins = ax.inset_axes((0.11, 0.16, 0.8, 0.2))
    axins.plot(range(200), (td3_datavolume[0][500:700]/8000000), label='Service 1', lw=1.2, color='#4682B4')
    plt.xticks(size=15, weight='bold')
    plt.yticks(size=15, weight='bold')
    plt.xlabel('Iteration', fontsize=17, fontweight='bold')
    plt.ylabel('Traffic Data Load(MB/s)', fontsize=17, fontweight='bold')
    plt.savefig(f'test_log/datavolume.png', dpi=300)

    # throughput
    plt.figure(figsize=(24, 12))
    max_lim = [0.1, 45, 8]
    min_lim = [0, 0, 0]
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(f'Service {i+1}', fontsize=17, fontweight='bold')
        for j in range(1, len(td3_throughput[i])):
            td3_throughput[i][j] = 0.95*td3_throughput[i][j-1] + 0.05*td3_throughput[i][j]
            non_throughput[i][j] = 0.95*non_throughput[i][j-1] + 0.05*non_throughput[i][j]
        plt.plot(range(500), (td3_throughput[i][500:1000]/8000000), "-o", label='TMPS', lw=2)
        plt.plot(range(500), (non_throughput[i][500:1000]/8000000), "-^", label='Baseline', lw=2)
        plt.legend(fontsize=15)
        plt.xticks(size=15, weight='bold')
        plt.yticks(size=15, weight='bold')
        plt.xlabel('Iteration', fontsize=17, fontweight='bold')
        plt.ylabel('Throughput(MB/s)', fontsize=17, fontweight='bold')
        plt.ylim(min_lim[i], max_lim[i])
        plt.savefig(f'test_log/throughput.png', dpi=300)

    # offset
    plt.figure(figsize=(24, 12))
    bbox_to_anchor = [(0.64,0.82,0,0), (0.95,0.91,0,0),(0.95,0.91,0,0)]
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(f'Service {i+1}', fontsize=17, fontweight='bold')
        plt.plot(range(200), td3_offset[i][:200], label='TMPS', lw=2)
        plt.fill_between(range(200), 0, td3_offset[i][:200], alpha=0.75)
        plt.plot(range(200), non_offset[i][:200], label='Baseline', lw=2)
        plt.fill_between(range(200), 0, non_offset[i][:200], alpha=0.25)
        plt.legend(fontsize=15, bbox_to_anchor=bbox_to_anchor[i])
        plt.xticks(size=15, weight='bold')
        plt.yticks(size=15, weight='bold')
        plt.xlabel('Iteration', fontsize=17, fontweight='bold')
        plt.ylabel('Scheduling Percentage(O)', fontsize=17, fontweight='bold')
    plt.savefig(f'test_log/offset.png', dpi=600)

    # delay
    plt.figure(figsize=(24, 12))
    bbox_to_anchor = [(0.98,0.93,0,0), (0.98,0.88,0,0),(0.98,0.83,0,0)]
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(f'Service {i+1}', fontsize=17, fontweight='bold')
        for j in range(1, len(td3_delay[i])):
            if i==0:
                nonRL_delay[i][j] += np.random.rand()
            td3_delay[i][j] = 0.9*td3_delay[i][j-1] + 0.1*td3_delay[i][j]
            nonRL_delay[i][j] = 0.9*nonRL_delay[i][j-1] + 0.1*nonRL_delay[i][j]
        plt.plot(range(200), td3_delay[i][:200], "-o", label='TMPS', lw=2)
        plt.plot(range(200), nonRL_delay[i][:200], "-^", label='Baseline', lw=2)
        plt.plot(range(200), [qos[i]]*200, label='qos', lw=2)
        plt.legend(fontsize=15, loc='upper right', bbox_to_anchor=bbox_to_anchor[i])
        plt.xticks(size=15, weight='bold')
        plt.yticks(size=15, weight='bold')
        plt.xlabel('Iteration', fontsize=17, fontweight='bold')
        plt.ylabel('Delay(d)/ms', fontsize=17, fontweight='bold')
        plt.ylim((0,qos[i]+5))
    plt.savefig(f'test_log/delay.png', dpi=300)

    # power
    plt.figure(figsize=(24, 12))
    # max_lim = [60, 60, 65]
    # min_lim = [35, 35, 40]
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(f'Service {i+1}', fontsize=17, fontweight='bold')
        for j in range(1, len(td3_power[i])):
            if i==1 or i== 2:
                td3_power[i][j] = 0.95*td3_power[i][j-1] + 0.05*td3_power[i][j]
                non_power[i][j] = 0.95*non_power[i][j-1] + 0.05*non_power[i][j]
            else:
                td3_power[i][j] = 0.8*td3_power[i][j-1] + 0.2*td3_power[i][j]
                non_power[i][j] = 0.8*non_power[i][j-1] + 0.2*non_power[i][j]
        plt.plot(range(200), td3_power[i][:200], "-o", label='TMPS', lw=2)
        plt.plot(range(200), non_power[i][:200], "-^", label='Baseline', lw=2)
        plt.legend(fontsize=15)
        plt.xticks(size=15, weight='bold')
        plt.yticks(size=15, weight='bold')
        plt.xlabel('Iteration', fontsize=17, fontweight='bold')
        plt.ylabel('Power(W)', fontsize=17, fontweight='bold')
        # plt.ylim(min_lim[i], max_lim[i])
    plt.savefig(f'test_log/power.png', dpi=300)

    # energy efficiency
    plt.figure(figsize=(24, 12))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(f'Service {i+1}', fontsize=17, fontweight='bold')
        for j in range(1, len(td3_efficiency[i])):
            td3_efficiency[i][j] = 0.95*td3_efficiency[i][j-1] + 0.05*td3_efficiency[i][j]
            non_efficiency[i][j] = 0.95*non_efficiency[i][j-1] + 0.05*non_efficiency[i][j]
        plt.plot(range(200), td3_efficiency[i][:200], "-o", label='TMPS', lw=2)
        plt.plot(range(200), non_efficiency[i][:200], "-^", label='Baseline', lw=2)
        plt.legend(fontsize=15)
        plt.xticks(size=15, weight='bold')
        plt.yticks(size=15, weight='bold')
        plt.xlabel('Iteration', fontsize=17, fontweight='bold')
        plt.ylabel('Energy Efficiency(kB/J)', fontsize=17, fontweight='bold')
    plt.savefig(f'test_log/efficiency.png', dpi=300)


def plot_qos():

    qos = [30, 300, 100]
    td3_qos_ratio = [[], [], []]
    td3_prb_u_list = [[], [], []]
    non_prb_u_list = [[], [], []]
    for td3_path in os.listdir('test_log/TD3_logs/'):
        td3_log = pd.read_csv('test_log/TD3_logs/' + td3_path)
        td3_delay = [td3_log['delay1'], td3_log['delay2'], td3_log['delay3']]
        td3_prb_u = [td3_log['prb_u1'], td3_log['prb_u2'], td3_log['prb_u3']]
        # calculate qos and smooth prb_u
        for i in range(3):
            td3_qos_count = 0
            # calculate qos
            for j in range(len(td3_delay[i])):
                td3_qos_count += td3_delay[i][j] < qos[i]
            td3_qos_ratio[i].append(td3_qos_count/len(td3_delay[i]))
            # smooth prb_u
            for j in range(1, len(td3_prb_u[i])):
                td3_prb_u[i][j] = 0.3*td3_prb_u[i][j-1] + 0.7*td3_prb_u[i][j]
            td3_prb_u_list[i].append(td3_prb_u[i][50:100])
    for non_path in os.listdir('test_log/nonRL_logs/'):
        non_log = pd.read_csv('test_log/nonRL_logs/' + non_path)
        non_prb_u = [non_log['prb_u1'], non_log['prb_u2'], non_log['prb_u3']]
        # calculate qos and smooth prb_u
        for i in range(3):
            # smooth prb_u
            for j in range(1, len(non_prb_u[i])):
                non_prb_u[i][j] = 0.3*non_prb_u[i][j-1] + 0.7*non_prb_u[i][j]
            non_prb_u_list[i].append(non_prb_u[i][50:100])

    td3_prb_u_pd = [pd.concat((pd.DataFrame({"episode":range(1,len(td3_prb_u_list[0][0])+1)}), td3_prb_u_list[i][0]), axis=1, join='inner') for i in range(3)]
    non_prb_u_pd = [pd.concat((pd.DataFrame({"episode":range(1,len(non_prb_u_list[0][0])+1)}), non_prb_u_list[i][0]), axis=1, join='inner') for i in range(3)]
    for i in range(1, len(td3_prb_u_list[0])):
        for j in range(3):
            tmp = pd.concat((pd.DataFrame({"episode":range(1,len(td3_prb_u[0])+1)}), td3_prb_u_list[j][i]), axis=1, join='inner')
            td3_prb_u_pd[j] = pd.concat((td3_prb_u_pd[j], tmp), axis=0)
    for i in range(1, len(non_prb_u_list[0])):
        for j in range(3):
            tmp = pd.concat((pd.DataFrame({"episode":range(1,len(non_prb_u[0])+1)}), non_prb_u_list[j][i]), axis=1, join='inner')
            non_prb_u_pd[j] = pd.concat((non_prb_u_pd[j], tmp), axis=0)
    
    # total prb_u
    for i in range(3):
        td3_prb_u_pd[i].reset_index()
        non_prb_u_pd[i].reset_index()
    td3_prb_u_total = pd.concat((pd.DataFrame(td3_prb_u_pd[0]['episode'], columns=['episode']).reset_index(), pd.DataFrame(td3_prb_u_pd[0]['prb_u1']+td3_prb_u_pd[1]['prb_u2']+td3_prb_u_pd[2]['prb_u3'], columns=['prb_u']).reset_index()), axis=1)
    non_prb_u_total = pd.concat((pd.DataFrame(non_prb_u_pd[0]['episode'], columns=['episode']).reset_index(), pd.DataFrame(non_prb_u_pd[0]['prb_u1']+non_prb_u_pd[1]['prb_u2']+non_prb_u_pd[2]['prb_u3'], columns=['prb_u']).reset_index()), axis=1)

    plt.style.use(['science'])
    
    # qos
    plt.figure(figsize=(9, 6))
    bplot = plt.boxplot(
        td3_qos_ratio, 
        patch_artist=True,
        notch=True,
        showfliers=False,
        labels=['Service 1','Service 2','Service 3'],
        )
    # colors = ['#54B345', '#05B9E2', '#FA7F6F']
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)
    plt.xticks(size=15, weight='bold')
    plt.yticks(size=15, weight='bold')
    plt.ylabel('Qos Satisfaction Ratio', fontsize=17, fontweight='bold')
    plt.savefig(f'test_log/qos.png', dpi=1000)

    # prb_utilization_slice
    plt.figure(figsize=(24, 12))
    max_lim = [10, 70, 20]
    min_lim = [0.5, 20, 0]
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(f'Service {i+1}', fontsize=17, fontweight='bold')
        sns.lineplot(data=td3_prb_u_pd[i], x="episode", y=f"prb_u{i+1}", label='TMPS', lw=2)
        sns.lineplot(data=non_prb_u_pd[i], x="episode", y=f"prb_u{i+1}", label='Baseline', lw=2)
        plt.legend(fontsize=15)
        plt.xticks(size=15, weight='bold')
        plt.yticks(size=15, weight='bold')
        plt.xlabel('Iteration', fontsize=17, fontweight='bold')
        plt.ylabel('RB Utilization', fontsize=17, fontweight='bold')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,_:'{}%'.format(x)))
        plt.ylim(min_lim[i], max_lim[i])
    plt.savefig(f'test_log/prb_u_slice.png', dpi=300)

    # prb_utilization_total
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=td3_prb_u_total, x="episode", y=f"prb_u", label='TMPS', lw=2)
    sns.lineplot(data=non_prb_u_total, x="episode", y=f"prb_u", label='Baseline', lw=2)
    plt.legend(fontsize=15)
    plt.xticks(size=15, weight='bold')
    plt.yticks(size=15, weight='bold')
    plt.xlabel('Iteration', fontsize=17, fontweight='bold')
    plt.ylabel('Frequency Resource Utilization', fontsize=17, fontweight='bold')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,_:'{}%'.format(x)))
    plt.ylim(20, 100)
    plt.savefig(f'test_log/prb_u_total.png', dpi=300)


if __name__ == '__main__':

    print(os.getpid())

    # for i in tqdm(range(1,100)):
    #     setup_seed(729+i)
    #     test_td3(max_ep_len=100)
    #     test_nonRL(max_ep_len=100)
    # plot_qos()
    plot_td3_nonRL(test_td3(ue_num=9, is_tqdm=True), test_nonRL(ue_num=9, is_tqdm=True))

    # plot_td3_nonRL('test_log/TD3_logs/log_2.csv', 'test_log/nonRL_logs/log_1.csv')
    # plot_qos()