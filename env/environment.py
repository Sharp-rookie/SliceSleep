# -*- coding: utf-8 -*-
import time
import numpy as np

from utils import setup_seed
from .BS import gNB


current_t = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

class args:
    def __init__(self):
        self.tti =1 # ms
        self.TD_schedule = 'TokenBucket'
        self.FD_schedule = 'RR'
        self.avg_interval = [50, 200, 300] # TTI
        self.avg_size = [1e5, 1e4, 5e5] # B
        self.ue_number = [5, 5, 5]
        self.seed = 729
        self.log_path = f'log/gnb_log_{current_t}.csv'


class Environment(object):
    """Environment for RL agent
    """

    def __init__(self, ue_number=None):
        
        arg = args()

        # 调整切片用户数量
        if ue_number:
            arg.ue_number = ue_number

        # 基站
        self.gnb = gNB(
            ue_number=arg.ue_number,
            TD_schedule=arg.TD_schedule,
            FD_schedule=arg.FD_schedule,
            avg_interval=[i/arg.tti for i in arg.avg_interval],
            avg_size=arg.avg_size,
            log_path=arg.log_path,
            tti=arg.tti,
            )

        # 设置随机种子
        setup_seed(arg.seed)
        
        # 确保3个bucket的rate相等
        assert(self.gnb.TD_policy.buckets[0].rate == self.gnb.TD_policy.buckets[1].rate == self.gnb.TD_policy.buckets[2].rate)

        # 统计量
        self.delay = [[], [], []] # 时延
        self.prb_utilization = [[], [], []] # PRB利用率
        self.throughput = [[], [], []] # 吞吐量
        self.power = [[], [], []] # 能耗
        self.datavolume = [[], [], []] # 数据量
        self.loss_pkt = [[], [], []] # 丢包数


    def step(self, action: list):
        """Step the environment
        """
        
        # 传入 TokenBucket 最新参数
        self.do_action(action)

        # 重置统计量
        self.delay, self.prb_utilization, self.throughput, self.power, self.datavolume = [None, None, None], [None, None, None], [None, None, None], [None, None, None], [None, None, None]

        # 仿真轮数
        simulate_rounds = 3

        self.simulate_duration = simulate_rounds*self.gnb.TD_policy.buckets[0].rate # 仿真TTI数必须是 token rate 的整数倍 ！！！

        # 按照新的参数仿真若干轮(TTI)基站
        self.gnb.reset() # 重置基站
        for _ in range(self.simulate_duration):
            self.gnb.simulate(print_log=False)
        self.gnb.statistic() # 更新基站统计量
            
        # 记录统计量
        for i in range(3):
            self.delay[i] = self.gnb.delay[i]
            self.prb_utilization[i] = self.gnb.prb_utilization[2][i]
            self.throughput[i] = self.gnb.throughput[i] / simulate_rounds /8/ self.gnb.TD_policy.buckets[0].rate # B/TTI
            self.datavolume[i] = self.gnb.datavolume[i] / simulate_rounds /8/ self.gnb.TD_policy.buckets[0].rate # B/TTI
            self.loss_pkt[i] = self.gnb.slice_loss[i] / simulate_rounds / self.gnb.TD_policy.buckets[0].rate #个/TTI
            self.power[i] = self.gnb.power[i] / simulate_rounds / self.gnb.TD_policy.buckets[0].rate # W/TTI

        state = self.get_state()
        reward = self.get_reward()

        done = [False] * 3
        for i, bucket in enumerate(self.gnb.TD_policy.buckets):
            if bucket.offset == 0:
                done[i] = True

        return state, reward, done

    def get_state(self):
        """Get the state of the environment
        """
        
        observation = []

        # Normalized DataVolume
        for i in range(3):
            mu = self.gnb.slice_ueNum[i] * (self.gnb.avg_size[i] / self.gnb.avg_interval[i])
            dataV = self.datavolume[i] / mu if self.datavolume[i] else 0.
            observation.append(dataV)

        [observation.append(self.gnb.TD_policy.buckets[i].offset) for i in range(3)]
        
        return observation


    def get_reward(self):
        """Get the reward of the environment
        """
        offset = [bucket.offset for bucket in self.gnb.TD_policy.buckets]
        qos_delay = [30, 300, 100] # eMBB, URLLC, mMTC

        reward = []
        for i in range(len(offset)):
            reward.append(1-offset[i] if self.gnb.delay[i]<qos_delay[i] and offset!=0 else 2*(offset[i]-1))
        
        return reward


    def do_action(self, action: list):
        """Do the action in the environment
        """
        
        assert(self.gnb.TD_policy.name == 'TokenBucket') # 只支持TokenBucket策略

        for i, bucket in enumerate(self.gnb.TD_policy.buckets):
            offset = round(bucket.offset + 0.025*action[i], 3)
            offset = max(0, offset)
            offset = min(1, offset)
            bucket.offset = offset # 四舍五入保留两位小数

    def render(self):
        """Render the environment
        """
        
        pass

    def reset(self):
        """Reset the environment
        """
        
        self.gnb.reset() # 重置原本基站

        for bucket in self.gnb.TD_policy.buckets:
            # bucket.offset = 1
            bucket.offset = 5*np.random.randint(1,20)/100
    
    def close(self):
        """Close the environment
        """

        self.gnb.close()