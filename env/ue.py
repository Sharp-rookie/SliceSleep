# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
from scipy.stats import uniform, poisson, expon

from .packet import Packet


class UE(object):
    """用户设备"""

    def __init__(self, id, type, tti=1, avg_interval=[6, 90, 80], avg_size=[5e4, 2e5, 1e5]):
        # 基本信息
        self.id = id                               # 基站侧id
        self.slice = type                          # 用户所属类型
        self.tti = tti                             # 单轮时长，ms
        self.snr = 700                             # 所在信道的信噪比，初始值700dBm
        self.pdcprate = 0                          # 单个PRB的发送量，反应信道质量，bit/s
        self.distance = np.random.randint(1, 161)  # 到基站距离，m
        self.request_list = deque()                # 请求列表
        self.received = []                         # 已收到但未清除的缓冲区数据包序号
        self.avg_size = avg_size                   # 请求包的平均大小，B（高斯分布）

        # 统计量
        self.delay = None                  # 平均时延，ms
        self.loss_pkt_num = 0             # 丢包数
        self.period_avg_prb = 0           # 统计周期内平均分得的PRB数量
        self.period_schedule_time = 0     # 统计周期内被调度次数
        self.total_request_pkt_num = 0    # 总请求数
        self.total_receive_pkt_num = 0    # 总接收数
        self.period_request_pkt_num = 0   # 统计周期内请求数
        self.period_receive_pkt_num = 0   # 统计周期内接收数

        # 请求到达间隔分布，TTI
        if self.slice == 'eMBB':
            self.interval_distribution = expon(loc=0, scale=avg_interval[0]/self.tti)
        elif self.slice == 'URLLC':
            self.interval_distribution = expon(loc=0, scale=avg_interval[1]/self.tti)
        elif self.slice == 'mMTC':
            self.interval_distribution = uniform(loc=0, scale=2*avg_interval[2]/self.tti)
        else:
            print('type error in UE.init()\n')
            exit()

        # 其他
        self.interval = 100        # 请求达到间隔，初始为100TTI ——> 用于判断是否来包
        self.duration = 0          # 距离上一次来包的时长，TTI ——> 用于判断是否来包
        self.requestPRB = 0        # 根据buffer计算的PRB需求
        self.prb = 0               # 分得的PRB个数
        self.removeCount = 0       # 记录TTI数量，每10000个TTI(5s)进行一次移动 ——> 用于随机移动
        self.wait = True           # 记录本轮是否一个包都没收到 ——> 用于计算delay
        self.throughput = 0        # 上一统计周期的吞吐量，bit  ——> 用于PF算法计算优先级
        self.buffersize = 0        # 缓冲区大小，bit  ——> 用于PF算法计算优先级

    def gen_request(self, buffer_index: int):
        """产生本业务的随机数据包请求

        Parameters:
        ------- 
           buffer_index: 该请求在基站buffer中的index

           duration: 当前TTI所在时刻，ms

        Return:
           pkts: 请求的pkt列表
        """

        # 判断包是否到达
        if self.duration < self.interval:
            self.duration += 1
            return None

        self.duration = 0 # 重置duration
                
        # 更新下一个包到达间隔
        self.interval = np.ceil(self.interval_distribution.rvs())

        # 进入用户请求列表
        if len(self.request_list) == 0:
            index1 = 0 # 该请求在UE侧的index
        else:
            index1 = self.request_list[-1].ue_index + 1
        pkt = Packet(self.id, index1, buffer_index, self.slice, self.avg_size)
        
        # 统计
        self.total_request_pkt_num += 1
        self.period_request_pkt_num += 1

        return pkt
    
    def randomMove(self):
        """用户随机移动
        """

        # 每1s移动一次
        self.removeCount = (self.removeCount + 1) % round(1*1000/self.tti)
        if self.removeCount != 0:
            return

        # 不同业务用户移动规律不同
        if self.slice == 'eMBB':
            self.distance += np.random.randint(-1, 2)
        elif self.slice == 'URLLC':
            self.distance += np.random.randint(-5, 6)
        elif self.slice == 'mMTC':
            pass
        else:
            raise TypeError(f'{self.slice} not implemented!')
        
        # 限制距离在0~160m之内
        self.distance = min(self.distance, 160)
        self.distance = max(self.distance, 1)
    
    def loss_pkt(self, packet: Packet):
        """因基站缓冲区满导致丢包
        """

        self.loss_pkt_num += 1
        del packet
    
    def receive_pkt(self, packet: Packet):
        """收到请求的数据包

        Parameters:
        ------- 
           ue_index: 该请求在UE侧的index
        """
        
        # 滑动平均QoE时延
        self.delay = 0.5*self.delay + 0.5*packet.delay if self.delay is not None else packet.delay

        # 记录已发送的pkt序号
        self.received.append(packet)

        # 统计
        self.total_receive_pkt_num += 1
        self.period_receive_pkt_num += 1
        self.throughput += packet.size
    
    def reset_statistic(self):
        """重置统计信息
        """

        self.delay = None
        self.period_receive_pkt_num = 0
        self.period_request_pkt_num = 0
        self.period_schedule_time = 0
        self.loss_pkt_num = 0
        self.period_avg_prb = 0

    def reset(self):
        """关闭用户，清除数据
        """

        # 清除缓冲区
        for pkt in self.request_list:
            del pkt
        self.request_list.clear()
        self.received = []

        # 重置统计数据
        self.delay = None                 # 平均时延，ms
        self.loss_pkt_num = 0             # 丢包数
        self.period_avg_prb = 0           # 统计周期内平均分得的PRB数量
        self.period_schedule_time = 0     # 统计周期内被调度次数
        self.total_request_pkt_num = 0    # 总请求数
        self.total_receive_pkt_num = 0    # 总接收数
        self.period_request_pkt_num = 0   # 统计周期内请求数
        self.period_receive_pkt_num = 0   # 统计周期内接收数

        # 重置其他量
        self.interval = 100        # 请求达到间隔，初始为100ms ——> 用于判断是否来包
        self.lastT = 0             # 上一次请求到达时刻 ——> 用于判断是否来包
        self.max_pkt_per_tti = 10  # 每个TTI的最大请求数量 ——> 用于时域调度
        self.requestPRB = 0        # 根据buffer计算的PRB需求
        self.prb = 0               # 分得的PRB个数
        self.removeCount = 0       # 记录TTI数量，每10000个TTI(5s)进行一次移动 ——> 用于随机移动
        self.wait = True           # 记录本轮是否一个包都没收到 ——> 用于计算delay
        self.throughput = 0        # 上一统计周期的吞吐量，bit  ——> 用于PF算法计算优先级
        self.buffersize = 0        # 缓冲区大小，bit  ——> 用于PF算法计算优先级

    
    def clear_satisfied_request(self):
        """缓冲区清除已收到的请求
        """

        if len(self.received) != 0:
            for pkt in self.received:
                self.request_list.remove(pkt)
        self.received = []