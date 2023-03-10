# -*- coding: utf-8 -*-
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})


def plot_result(path, test_dir):
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