# -*- coding: utf-8 -*-
import numpy as np

from utils import setup_seed
from .BS import BaseStation


class args:
    def __init__(self):
        self.tti = 1.  # ms
        self.avg_interval = [50, 10, 100]  # TTI
        self.avg_size = [300*1e3, 0.1*1e3, 40*1e3]  # B
        self.ue_number = [5, 5, 5]
        self.seed = 729
        self.bucket_config = [[160,1.], [160,1.], [160,1.]]  # [period, wakeup_ratio]
        self.action_space = [-0.025, 0, 0.025]


class Environment(object):
    """Environment for RL agent"""

    def __init__(self, ue_number=None):
        
        arg = args()

        # set the UE number of each slice
        if ue_number:
            arg.ue_number = ue_number

        # init BS
        self.BS = BaseStation(
            ue_number=arg.ue_number,
            avg_interval=[i/arg.tti for i in arg.avg_interval],
            avg_size=arg.avg_size,
            tti=arg.tti,
            bucket_config=arg.bucket_config
            )
        
        self.tti = arg.tti
        self.sim_duration = 5*self.BS.TD_policy.buckets[0].period # simulation duration, TTI
        self.action_space = arg.action_space
        
        # statistic
        self.delay = [[] for _ in range(3)]
        self.datavolume = [[] for _ in range(3)]
        self.throughput = [[] for _ in range(3)]
        self.prb_utilization = 0.
        self.fixed_consumption = 0.
        self.load_consumption = [[] for _ in range(3)]
        self.switch_consumption = [[] for _ in range(3)]

        setup_seed(arg.seed)

    def do_action(self, action: list):
        """Adjust the sleep settings of the BS"""
        
        for i, bucket in enumerate(self.BS.TD_policy.buckets):
            wakeup_ratio = round(bucket.wakeup_ratio + self.action_space[action[i]], 3)
            wakeup_ratio = max(0, wakeup_ratio)
            wakeup_ratio = min(1, wakeup_ratio)
            bucket.wakeup_ratio = wakeup_ratio

    def get_state(self):
        """Get environment state"""

        # data volume
        datavolume = []
        for i in range(3):
            # normalize
            mu = self.BS.slice_ueNum[i] * (self.BS.avg_size[i] / self.BS.avg_interval[i]) * (1000/self.tti) # bit/s
            data_vol = self.datavolume[i] / mu
            
            datavolume.append(data_vol)

        # state
        state = [[] for _ in range(3)]
        for i in range(3):
            # state[i].append(np.sum(datavolume))
            # state[i].append(datavolume[i])
            # [state[i].append(self.BS.TD_policy.buckets[j].wakeup_ratio) for j in range(3)] # wakeup_ratio
            state[i].append(self.BS.TD_policy.buckets[i].wakeup_ratio) # wakeup_ratio

        return state

    
    def step(self, action: list):
        """Interact with the environment"""
        
        # do action
        self.do_action(action)

        # simulating
        self.BS.reset()
        for _ in range(self.sim_duration):
            self.BS.simulate()
        self.BS.statistic()

        # statistic
        for i in range(3):
            self.delay[i] = self.BS.delay[i]
            self.datavolume[i] = self.BS.datavolume[i] / self.sim_duration * (1000/self.tti) # bit/s
            self.throughput[i] = self.BS.throughput[i] / self.sim_duration * (1000/self.tti) # bit/s
            self.prb_utilization = self.BS.prb_utilization
            self.fixed_consumption = self.BS.fixed_consumption / self.sim_duration * (1000/self.tti) # J / s
            self.load_consumption[i] = self.BS.load_consumption[i] / self.sim_duration * (1000/self.tti) # J / s
            self.switch_consumption[i] = self.BS.switch_consumption[i] / self.sim_duration * (1000/self.tti) # J / s

        # state
        state = self.get_state()

        # reward
        qos_delay = [100, 20, 300] # eMBB, URLLC, mMTC
        reward = [[] for _ in range(3)]
        dones = [False for _ in range(3)]
        for i in range(3):
            if self.delay[i] > qos_delay[i]:
                reward[i] = (qos_delay[i] - self.delay[i])
            else:
                max_consumption = self.BS.fixed_power_wake + self.BS.load_power  # W*s  # TODO: 负载量未定义
                real_consumption = self.fixed_consumption + self.load_consumption[i]
                power_saving = max_consumption - real_consumption - self.switch_consumption[i]
                reward[i] = power_saving
            
            if self.delay[i] > qos_delay[i]:
                reward[i] = [-1,0,1][action[i]]
            else:
                reward[i] = [1,0,-1][action[i]]
            
            if self.BS.TD_policy.buckets[i].wakeup_ratio == 0.:
                dones[i] = True

        return state, reward, dones

    def reset(self):
        """Reset the environment"""
        
        self.BS.reset()  # reset BS

        # random init sleep duration ratio
        for bucket in self.BS.TD_policy.buckets:
            bucket.wakeup_ratio = 1.
    
    def close(self):
        """Close the environment"""
        self.BS.close

#     def print_log(self):

#         print('-------------------------------------------------------------')
#         print([self.BS.TD_policy.buckets[i].wakeup_ratio for i in range(3)])
#         print(self.delay, "ms")
#         print([round(self.datavolume[i]) for i in range(3)], "b")
#         print([round(self.throughput[i]) for i in range(3)], "b")
#         print(round(self.prb_utilization*100), "%")
#         print(round(self.fixed_consumption), "W*s")
#         print([round(self.load_consumption[i]) for i in range(3)], "W*s")
#         print(self.switch_consumption, "J")


# if __name__ == '__main__':

#     env = Environment(ue_number=[5]*3)
#     for _ in range(90):
#         env.step([3,3,3])
#         env.print_log()