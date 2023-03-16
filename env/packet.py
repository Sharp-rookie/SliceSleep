# -*- coding: utf-8 -*-

currentTTI = 0  # 系统当前TTI，用于全局定时（与实际时间不同，因为实际时间受代码运行效率和硬件性能影响）

class Packet(object):
    """数据包请求"""
    def __init__(self, ueid, index1, index2, type, avg_size=[5e4, 2e5, 1e5]):
        self.ueid = ueid                       # 所属ueid
        self.slice = type                      # 所属业务类型
        self.ue_index = index1                 # 在UE侧的id
        self.buffer_index = index2             # 在基站缓冲区的id
        self.prb = 0                           # 分得prb个数
        self.startT = currentTTI               # 产生时刻的时间戳，TTI
        self.delay = 0                         # 时延等于排队时延 + 传输时延，ms

        # 数据包大小分布
        # self.embb_distribution = norm(loc=8*avg_size[0], scale=0.1) # 均值为 ，方差为0.1的正态分布
        # self.urllc_distribution = norm(loc=8*avg_size[1], scale=0.7)
        # self.mmtc_distribution = norm(loc=8*avg_size[2], scale=0)
        self.embb_size = 8*avg_size[0]
        self.urllc_size = 8*avg_size[1]
        self.mmtc_size = 8*avg_size[2] # 常数

        self.size = self.randomSize(type) # 数据包大小，bit
    
    def randomSize(self, type: str):
        """按照业务类型，随机指定大小数据包，单位：B
        """

        if type == 'eMBB':
            # return int(self.embb_distribution.rvs())
            return int(self.embb_size)
        elif type == 'URLLC':
            return int(self.urllc_size)
        elif type == 'mMTC':
            return int(self.mmtc_size)
        else:
            raise TypeError(f'{type} not implemented!\n')