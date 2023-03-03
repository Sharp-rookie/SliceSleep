# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from collections import deque
from scipy.stats import uniform, poisson, expon

from utils import bcolors, Visualizer
from .FD_schedule import RR_slice_FD, RR_FD, PF_FD
from .TD_schedule import TokenBucket, RR_select

__all__ = [
    'gNB',
    ]

currentTTI = 0  # 系统当前TTI，用于全局定时（与实际时间不同，因为实际时间受代码运行效率和硬件性能影响）

# 打点计时
t_stamp = [round(1000*time.time())] * 5 # request_t, td_schedule_t, print_t, fd_schedule_t, send_t
duration = [0, 0, 0, 0]


bucket_config = [[300,33,0.95], [300,33,0.75], [300,33,0.9]] # token bucket init config [token rate, reservedPRB, offset]
prb_config = [33, 33, 33] # prb config for RR_select TD scheduling


def dBm2W(dBm):
    """Convert dBm to W
    """
    return 10**(dBm/10) / 1000


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
        self.embb_distribution = poisson(mu=8*avg_size[0])
        self.urllc_size = 8*avg_size[1]
        self.mmtc_size = 8*avg_size[2] # 常数

        self.size = self.randomSize(type) # 数据包大小，bit
    
    def randomSize(self, type: str):
        """按照业务类型，随机指定大小数据包，单位：B
        """

        if type == 'eMBB':
            return int(self.embb_distribution.rvs())
        elif type == 'URLLC':
            return int(self.urllc_size)
        elif type == 'mMTC':
            return int(self.mmtc_size)
        else:
            print('type error in Packet.randomSize()\n')
            exit()


class UE(object):
    """用户设备"""

    def __init__(self, id, type, tti=1, avg_interval=[6, 90, 80], avg_size=[5e4, 2e5, 1e5]):
        # 基本信息
        self.id = id                               # 基站侧id
        self.slice = type                          # 用户所属类型
        self.tti = tti                             # 单轮时长，ms
        self.snr = 700                             # 所在信道的信噪比，初始值700dBm
        self.pdcprate = 0                          # 单个PRB的发送量，反应信道质量，bit/s
        self.distance = np.random.randint(1, 51)  # 到基站距离，m
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

        # 每5s移动一次
        self.removeCount = (self.removeCount + 1) % round(5*1000/self.tti)
        if self.removeCount != 0:
            return

        # 不同业务用户移动规律不同
        if self.slice == 'eMBB':
            self.distance += np.random.randint(-1, 2)
        elif self.slice == 'URLLC':
            self.distance += np.random.randint(-1, 2)
        elif self.slice == 'mMTC':
            self.distance += np.random.randint(-1, 2)
        else:
            print('type error in UE.randomMove()\n')
            exit()
        
        # 限制距离在0~160m之内
        self.distance = min(self.distance, 50)
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
        


class gNB(object):
    """基站"""

    MAX_ACTIVE_UE = 1e5 # per TTI 最大活跃用户数量，目前不设限
    
    global currentTTI
    currentTTI = 0 # 系统当前的TTI

    def __init__(self,
        ue_number: list,
        TD_schedule='RR_select',
        RL_agent=None,
        FD_schedule='RR',
        avg_interval=[6, 90, 80],
        avg_size = [5e4, 2e5, 1e5],
        log_path='log/gnb_log.csv',
        tti = 1,
        write=False,
        plot=False,
        Print=False,
        print_period=3
        ):

        # 基本信息
        self.tti = tti                           # TTI时长，ms
        self.slices = ['eMBB', 'URLLC', 'mMTC']  # 服务的业务类型
        self.UEs = []                            # 用户列表
        self.slice_ueNum = ue_number             # 各业务UE数量
        self.ueNum = 0                           # 用户数量
        self.pkt_buffer = deque()                # 切片公用的请求缓冲区
        self.BUFFERSIZE = 3000                   # 缓冲区大小
        self.totalPRB = 100                      # 可用prb总数
        self.bandPRB = 2e5                       # per PRB 带宽，hz
        self.transP = 46                         # 发射功率，dBm
        self.AWGN = -174                         # 加性高斯白噪声功率，dBm/hz
        self.avg_size = avg_size                 # 请求包的平均大小，B（高斯分布）
        self.avg_interval = avg_interval         # 请求包的平均间隔，s（高斯分布）

        # 统计量
        self.print_activate_ue = []                          # 每个TTI打印的活跃用户，其实无意义，只是为了好看
        self.write = write                                   # 是否写入log
        self.plot = plot                                     # 是否画图
        self.print = Print                                   # 是否打印log
        self.print_period = print_period                     # 打印log的周期，单位为s
        self.log_f = open(log_path,"w+") if write else None  # 日志文件
        self.log_path = log_path                             # 日志文件路径
        self.print_t = print_period*1000                     # 用于打印计时
        self.start_t = round(1000*time.time())               # 开始时刻，ms
        self.total_request_pkt_num = 0                       # 记录已接收请求包总数
        self.total_send_pkt_num = 0                          # 记录已发送包总数
        self.total_loss_pkt_num = 0                          # 记录丢包总数
        self.slice_request = [0, 0, 0]                       # 每个业务周期请求量
        self.slice_send = [0, 0, 0]                          # 每个业务周期发送量
        self.slice_loss = [0, 0, 0]                          # 每个业务周期丢包量
        self.delay = [999, 999, 999]                         # 三个业务的时延
        self.datavolume = [0, 0, 0]                          # 三个业务的请求量
        self.throughput = [0, 0, 0]                          # 三个业务的吞吐量
        self.power = [0, 0, 0]                               # 三个业务的能耗
        
        # PRB: 使用量、预留量、利用率，目前prb利用率只反映频域资源情况，具体调度效果需要结合时域调度机会一起分析
        self.prb_utilization = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  
        
        # 时域调度策略
        if TD_schedule == 'RR_select':
            self.TD_policy = RR_select(prb_config)
        elif TD_schedule == 'TokenBucket':
            self.TD_policy = TokenBucket(bucket_config) 
        self.sleep = [False] * 3 # 休眠
            
        # RL agent
        if RL_agent == 'PPO':
            from RL_policy.ppo import PPO
            self.RL_agent = PPO
        elif RL_agent == 'TD3':
            from RL_policy.td3 import TD3
            self.RL_agent = TD3
        elif RL_agent == 'DQN':
            from RL_policy.dqn import DQN
            self.RL_agent = DQN
        else:
            self.RL_agent = None

        # 频域调度策略 
        if FD_schedule == 'RR_slice':
            self.FD_policy = RR_slice_FD()
        elif FD_schedule == 'RR':
            self.FD_policy = RR_FD()
        elif FD_schedule == 'PF':
            self.FD_policy = PF_FD()

        # 其他
        self.start_time = 0        # 每轮的开始时刻，ms
        self.activeUEs = []        # 挑选为的活跃用户，最大20个
        self.lastUE_id = 0         # 即将被调度UE的id，RR的方式进行时域调度

        # 生成UE
        if len(self.slice_ueNum) != 3:
            print("Invalid ue number!\n")
            assert(False)
        self.__initUEs(self.slice_ueNum)

        # 日志文件标题
        if self.write:
            self.log_f.write('time,power,throughput,datavolume,buffer,embb_delay,urllc_delay,mmtc_delay,embb_prb_u,urllc_prb_u,mmtc_prb_u\n')
    
        # 可视化端口
        if self.plot:
            # os.system("gnome-terminal -e 'python -m visdom.server -port 8097'")
            self.vis = Visualizer(env='GNB', port=8097)

    def __initUEs(self, number: list):
        """生成并初始化UE
                
        Parameters:
        ------- 
           number: 各类型用户数量, [embb urllc mmtc]
        """

        if any(number) < 0:
            print("Invalid ue number!\n")
            assert(False)
        print(number)

        assert(self.UEs == [])
        for i in range(number[0]):
            self.UEs.append(UE(i, 'eMBB', self.tti, self.avg_interval, self.avg_size))
        for i in range(number[1]):
            self.UEs.append(UE(i+number[0], 'URLLC', self.tti, self.avg_interval, self.avg_size))
        for i in range(number[2]):
            self.UEs.append(UE(i+number[0]+number[1], 'mMTC', self.tti, self.avg_interval, self.avg_size))
        
        self.ueNum = len(self.UEs)
    
    def simulate(self, print_log=True):
        """仿真一个TTI的调度全过程
        """

        self.update_ue_request() # 用户请求到达
        self.update_ue_channel_quality() # 更新用户信道质量
        self.sleep = self.select_active_ue() # 时域调度
        self.allot_prb() # 频域调度
        self.send_ue() # 发包
        if print_log:
            self.print_log() # 打印日志
    
    def update_ue_request(self):
        """遍历所有用户，按业务类型随机产生请求并计入缓冲
        """

        # TODO: 这里运行效率低，时延在 0~10ms 左右

        # 计时
        t_stamp[0] = round(1000*time.time())

        # 更新系统当前TTI
        global currentTTI
        currentTTI += 1
        
        # 记录轮开始时刻
        self.start_time = round(1000*time.time())
        
        # 轮询访问请求
        for ue in self.UEs:
            if len(self.pkt_buffer) == 0:
                buffer_index = 0 
            else:
                buffer_index = self.pkt_buffer[-1].buffer_index + 1
            
            pkt = ue.gen_request(buffer_index)
            if pkt == None:
                continue
            else:
                if len(self.pkt_buffer) >= self.BUFFERSIZE:
                    ue.loss_pkt(pkt)
                    self.total_loss_pkt_num += 1
                    self.slice_loss[self.slices.index(ue.slice)] += 1
                else:
                    ue.request_list.append(pkt)
                    self.pkt_buffer.append(pkt)
                
                self.total_request_pkt_num += 1
                self.datavolume[self.slices.index(ue.slice)] += pkt.size
                self.slice_request[self.slices.index(ue.slice)] += 1
            
    def cal_prb_request(self, ue: UE):
        """根据UE缓冲区的请求数据包总大小和信道质量计算所需prb个数
        """

        if len(ue.request_list) == 0:
            print('UE buffer size == 0!')
            return

        nTBSize = ue.pdcprate * (self.tti/1000)

        # update buffersize
        buffersize = 0
        for pkt in ue.request_list:
            buffersize += pkt.size
        ue.buffersize = buffersize
        
        ue.requestPRB = int(np.ceil(ue.buffersize / nTBSize))

    def update_ue_channel_quality(self):
        """计算UE的信道质量 pdcprate
        """

        for ue in self.UEs:
            ue.randomMove() # 更新用户随机移动后的位置

            # https://www.cnblogs.com/jobgeo/p/5202625.html#:~:text=%E8%87%AA%E7%94%B1%E7%A9%BA%E9%97%B4%E6%8D%9F%E8%80%97%E6%98%AF%E6%8C%87%E7%94%B5%E7%A3%81%E6%B3%A2%E5%9C%A8%E4%BC%A0%E8%BE%93%E8%B7%AF%E5%BE%84%E4%B8%AD%E7%9A%84%E8%A1%B0%E8%90%BD%2C%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F%E5%A6%82%E4%B8%8B%EF%BC%9A,Lbf%3D32.5%2B20lgF%2B20lgD
            # https://www.zhihu.com/question/32326772
            Lbf = 31.5 + 35*np.log10(ue.distance/1000 + 1e-3) # 路径损耗
            noiseP = self.AWGN * (ue.prb * self.bandPRB) # 噪声功率
            ue.snr = dBm2W(self.transP - Lbf) / (dBm2W(noiseP) + 1e-3)

            pdcprate = self.bandPRB * np.log2(1 + ue.snr)
            ue.pdcprate = pdcprate

    def select_active_ue(self):
        """时域调度
        """

        # 计时
        t_stamp[1] = round(1000*time.time())

        # RL 接口
        if self.TD_policy.name == 'TokenBucket':
            # self.RL_agent.update_token_bucket()
            pass

        # 按照封装好的算法，选择激活的用户
        return self.TD_policy.schedule(self, self.prb_utilization) # TODO: 时域调度预留PRB的部分还没做，目前是取固定值
                
    def checkUEBufferInfo(self):
        """检查用户当前是否有数据包请求
        """

        if len(self.activeUEs) == 0:
            return False
        
        # 根据用户的请求缓冲区，计算用户所需prb数量，缓冲区为空则出队
        remove_ue = []
        for ue in self.activeUEs:
            if len(ue.request_list) == 0:
                remove_ue.append(ue)
            else:
                self.cal_prb_request(ue)
        
        if len(remove_ue)!= 0:
            [self.activeUEs.remove(ue) for ue in remove_ue]
        del remove_ue
    
    def allot_prb(self):
        """频域调度
        """

        # 计时
        t_stamp[2] = round(1000*time.time())

        # 根据用户请求数据包的量，计算所需prb个数
        self.checkUEBufferInfo()

        # 按照封装好的算法，分配prb
        self.FD_policy.schedule(self.TD_policy.reservedPRB, self.activeUEs, self.prb_utilization)

    def send_ue(self):
        """为UE发包
        """

        # 计时
        t_stamp[3] = round(1000*time.time())

        # DU负载损耗
        load_ratio = [0] * 3 # 切片调度负载量
        for ue in self.activeUEs:
            if ue.slice == 'eMBB':
                load_ratio[0] += 1
            if ue.slice == 'mMTC':
                load_ratio[1] += 1
            if ue.slice == 'URLLC':
                load_ratio[2] += 1
        for i in range(3):
            # self._power[i] += 100 * (self._tti/1000) # 基站电路功耗
            if not self.sleep[i]: # DU切片处理功耗
                load_ratio[i] = load_ratio[i] / self.MAX_ACTIVE_UE * 3 # 乘3保证负载总和为100%
                self.power[i] += 50 * (self.tti/1000) # DU空闲功耗
                self.power[i] += load_ratio[i]*(100-50) * (self.tti/1000) # DU负载功耗

        # 按照分得的prb进行发包
        for ue in self.activeUEs:
            if ue.prb > 0: # 原本是prb大于requestPRB才发送，相当于把用户的请求全发完；现在改成能发多少发多少
                self.send_pkt(ue)
            else:
                pass
        if len(self.activeUEs):
            self.print_activate_ue = [ue.id for ue in self.activeUEs]
        self.activeUEs = [] # 为了保证公平性，即使UE没有发送完请求，也不能继续留在活跃列表中

    def send_pkt(self, ue: UE):
        """发出该用户请求的数据包或数据包的一部分
        """

        if len(ue.request_list) == 0:
            return
        
        sliceid = self.slices.index(ue.slice)
        available_throughput = ue.prb * ue.pdcprate * (self.tti/1000) # 目前所分prb在一个TTI内的发送能力
        for ue_pkt in ue.request_list:
            if available_throughput > 0:
                if ue_pkt.size <= available_throughput:
                    # 计算时延
                    queue_delay = (currentTTI - ue_pkt.startT) * self.tti # 排队时延
                    transmit_delay = ue_pkt.size / (ue.prb * ue.pdcprate + 1e-3) * 1000 # 发送时延，单位ms
                    ue_pkt.delay = queue_delay + transmit_delay

                    # 从基站buffer中移出
                    for gnb_pkt in self.pkt_buffer:
                        if gnb_pkt.buffer_index == ue_pkt.buffer_index:
                            self.pkt_buffer.remove(gnb_pkt)
                            break
                        
                    # UE接收pkt
                    ue.receive_pkt(ue_pkt)

                    # 消耗发送能力
                    available_throughput -= ue_pkt.size

                    # 统计
                    self.total_send_pkt_num += 1 # 记录已发送包总数
                    self.slice_send[sliceid] += 1
                    self.throughput[sliceid] += ue_pkt.size # 发送量

                    # 删除数据包
                    del ue_pkt

                    # 确认本轮收到包
                    ue.wait = False
                else:
                    # 先只发送包的一部分
                    ue_pkt.size -= available_throughput

                    self.throughput[sliceid] += available_throughput

                    available_throughput = 0
            else:
                break
        
        ue.prb = 0 # UE分得的prb清零，保证每轮prb不能复用
        ue.clear_satisfied_request()
    
    def reset(self):
        """重置GNB，每次更换资源分配策略时调用
        """

        # 重置 UE
        for ue in self.UEs:
            ue.reset()

        # 清空缓冲区
        self.pkt_buffer.clear()

        # 重置统计信息
        self.start_t = round(1000*time.time())               # 开始时刻，ms
        self.print_t = self.print_period*1000                # 用于打印计时
        self.total_request_pkt_num = 0                       # 记录已接收请求包总数
        self.total_send_pkt_num = 0                          # 记录已发送包总数
        self.total_loss_pkt_num = 0                          # 记录丢包总数
        self.slice_request = [0, 0, 0]                       # 每个业务周期请求量
        self.slice_send = [0, 0, 0]                          # 每个业务周期发送量
        self.slice_loss = [0, 0, 0]                          # 每个业务周期丢包量
        self.delay = [None, None, None]                      # 三个业务的时延
        self.power = [0, 0, 0]                               # 三个业务的功耗
        self.datavolume = [0, 0, 0]                          # 三个业务的请求量
        self.throughput = [0, 0, 0]                          # 三个业务的吞吐量
        self.prb_utilization = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # PRB: 使用量、预留量、利用率

        # 重置其他信息
        self.activeUEs = []        # 挑选为的活跃用户，最大20个
        self.lastUE_id = 0         # 即将被调度UE的id，RR的方式进行时域调度
    
    def reshape_traffic(self, avg_interval, avg_size):
        """重置流量模式
        """
        
        self.reset()
        for ue in self.UEs:
            del ue
        
        self.UEs = []
        self.avg_interval, self.avg_size = avg_interval, avg_size
        self.__initUEs(self.slice_ueNum)
        
    
    def close(self):
        """关闭基站仿真
        """

        # 清除数据
        for ue in self.UEs:
            for pkt in ue.request_list:
                del pkt
            del ue
        
        # 关闭文件
        if self.log_f:
            print(f'Save log file at {self.log_path}')
            self.log_f.close()

    def statistic(self):

        # 计时
        t_stamp[4] = round(1000*time.time())

        # 计算各阶段时长
        duration[0] = 0.5*duration[0] + 0.5*(t_stamp[1] - t_stamp[0]) # 更新用户请求
        duration[1] = 0.5*duration[1] + 0.5*(t_stamp[2] - t_stamp[1]) # 时域调度
        duration[2] = 0.5*duration[2] + 0.5*(t_stamp[3] - t_stamp[2]) # 频域调度
        duration[3] = 0.5*duration[3] + 0.5*(t_stamp[4] - t_stamp[3]) # 发包
        
        # delay
        tmp_delay = [0, 0, 0]
        for ue in self.UEs:
            if ue.wait: # 本轮UE未收到任何包
                ue.delay = 999
            ue.wait = True
            slice_id = self.slices.index(ue.slice)
            tmp_delay[slice_id] += ue.delay
        for slice_id in range(3):
            tmp_delay[slice_id] /= self.slice_ueNum[slice_id] + 1e-3
            
            if self.delay[slice_id] is not None and self.delay[slice_id] < 500: # 小于500ms时，取滑动平均
                self.delay[slice_id] = 0.3*self.delay[slice_id] + 0.7*tmp_delay[slice_id]
            else:
                self.delay[slice_id] = tmp_delay[slice_id]
        
        # prb utilization
        for i in range(3):
            self.prb_utilization[2][i] = 100 * self.prb_utilization[0][i] / (self.prb_utilization[1][i] + 1e-3)
            self.prb_utilization[0][i] = 0
            self.prb_utilization[1][i] = 0
        
        # ue avg prb
        for ue in self.UEs:
            ue.period_avg_prb = int(ue.period_avg_prb/(ue.period_schedule_time + 1e-3))
    
    def reset_statistic(self):
        """重置统计信息
        """

        # UE
        for ue in self.UEs:
            ue.reset_statistic()

        # GNB
        self.throughput = [0, 0, 0]
        self.datavolume = [0, 0, 0]
        self.power = [0, 0, 0]
        self.slice_request = [0, 0, 0]
        self.slice_send = [0, 0, 0]
        self.slice_loss = [0, 0, 0]

    
    def print_log(self):

        # 打印定时
        tmp = 1000*time.time() - self.start_t
        if not (tmp >= self.print_t):
            return 
        self.print_t += self.print_period*1000

        # 更新统计量
        self.statistic() 

        # 写入日志
        if self.write:
            self.log_f.write(f'{round(tmp/1000)},{sum([self.throughput[i]/1e6 for i in range(3)])},{sum([self.datavolume[i]/1e6 for i in range(3)])},{len(self.pkt_buffer)},{round(self.delay[0],2)},{round(self.delay[1],2)},{round(self.delay[2],2)},{self.prb_utilization[2][0]},{self.prb_utilization[2][1]},{self.prb_utilization[2][2]}\n')
            self.log_f.flush()

        # 可视化
        if self.plot:
            for i in range(len(self.slices)):
                self.vis.plot(win='Delay', name=self.slices[i], y=self.delay[i])
                self.vis.plot(win='Prb_Utilization', name=self.slices[i], y=self.prb_utilization[2][i])
                self.vis.plot(win=f'DataVolume & Throughput ({self.slices[i]})', name='throughput', y=self.throughput[i])
                self.vis.plot(win=f'DataVolume & Throughput ({self.slices[i]})', name='datavolume', y=self.datavolume[i])

        # 打印统计量
        if self.print:
            # Title
            os.system('clear') if os.name=='posix' else os.system('cls')
            print(f'{bcolors.gray}-------------------------------------------------------------------------------------------------------------------------------------{bcolors.end}')
            print(f'{bcolors.bold}{bcolors.green}Dian{bcolors.end} {bcolors.bold}{bcolors.blue}Intel{bcolors.end} {bcolors.bold}{bcolors.red}NS{bcolors.end} {bcolors.bold}{bcolors.purple}Simulator{bcolors.end}'.center(182))
            
            # System Information
            print(f'{bcolors.gray}-------------------------------------------------------------------------------------------------------------------------------------{bcolors.end}')
            print(f'{bcolors.bold}System Information (per {self.print_period}s){bcolors.end}')
            
            print(' '.ljust(9), 'UE Request'.center(14), 'TD Schedule'.center(14), 'FD Schedule'.center(14), 'Send Pkt'.center(14), 'TTI'.center(6), 'Current TTI'.center(14), 'Total Time'.center(14))
            print(f'{bcolors.bold}Time{bcolors.end}'.ljust(17), f'{round(duration[0])} ms'.center(14), f'{round(duration[1])} ms'.center(14), f'{round(duration[2])} ms'.center(14),
             f'{round(duration[3])} ms'.center(14), f'{self.tti} ms'.center(6), f'{currentTTI}'.center(14), f'{bcolors.bold}{round(time.time()-self.start_t/1000, 2)} s{bcolors.end}'.center(23))
            
            print()
            print(f'                     ', f'{bcolors.bold}eMBB{bcolors.end}'.center(30), f'{bcolors.bold}URLLC{bcolors.end}'.center(33), f'{bcolors.bold}mMTC{bcolors.end}'.center(32))
            print(f'Packet Avg Interval: ', f'expon(mu={self.avg_interval[0]} ms)'.center(24), f'expon(mu={self.avg_interval[1]} ms)'.center(24), f'uniform(mu={self.avg_interval[2]} ms)'.center(24))
            print(f'Packet Avg Size:     ', f'possion(mu={self.avg_size[0]} B)'.center(24), f'{int(self.avg_size[1])} B'.center(24), f'{self.avg_size[2]} B'.center(24))
            
            if self.plot:
                print(f'Visualize link:      ', 'http://localhost:8097', f' (plot per {self.print_period} s)')
            
            # GNB Information
            print(f'\n{bcolors.gray}------------------------------------------------------------- gNB -------------------------------------------------------------------{bcolors.end}')
            print(' '.center(6), f'{bcolors.bold}Num{bcolors.end}'.center(13), f'{bcolors.bold}Delay{bcolors.end}'.center(18), f'{bcolors.bold}PRB_Util{bcolors.end}'.center(17),
             f'{bcolors.bold}Throughput{bcolors.end}'.center(21), f'{bcolors.bold}Datavolume{bcolors.end}'.center(19), f'{bcolors.bold}RequestNum{bcolors.end}'.center(21),
              f'{bcolors.bold}SendNum{bcolors.end}'.center(19), f'{bcolors.bold}LossNum{bcolors.end}'.center(19), f'{bcolors.bold}TokenRate{bcolors.end}'.center(12),
              f'{bcolors.bold}reservedPRB{bcolors.end}'.center(22), f'{bcolors.bold}Offset{bcolors.end}'.center(10))
            
            for i in range(3):
                print(f'{self.slices[i]}'.ljust(6), f'{self.slice_ueNum[i]}'.center(5), f'{bcolors.blue}{round(self.delay[i],2)} ms{bcolors.end}'.center(19),
                 f'{bcolors.yellow}{round(self.prb_utilization[2][i],2)}%{bcolors.end}'.center(19), f'{round(self.throughput[i]/1e6,2)} Mb'.center(13), f'{round(self.datavolume[i]/1e6,2)} Mb'.center(12),
                  f'{self.slice_request[i]}'.center(11), f'{bcolors.green}{self.slice_send[i]}{bcolors.end}'.center(20), f'{bcolors.red}{self.slice_loss[i]}{bcolors.end}'.center(20),
                   f'{self.TD_policy.buckets[i].rate}ms'.center(10), f'{self.TD_policy.buckets[i].reservedPRB}'.center(11), f'{100*self.TD_policy.buckets[i].offset}%'.center(10))
            
            print()
            print(f'{bcolors.bold}Request:{bcolors.end}'.ljust(13), f'{self.total_request_pkt_num}'.ljust(12), f'{bcolors.bold}Sended:{bcolors.end}'.ljust(13), f'{self.total_send_pkt_num}'.ljust(10),
             f'{bcolors.bold}FD Schedule:{bcolors.end}'.ljust(13), f'{bcolors.purple}{self.FD_policy.name}{bcolors.end}')
            print(f'{bcolors.bold}Buffer :{bcolors.end}'.ljust(13), f'{bcolors.green}{len(self.pkt_buffer)}/{self.BUFFERSIZE}{bcolors.end}'.ljust(21), f'{bcolors.bold}Lossed:{bcolors.end}'.ljust(13),
             f'{bcolors.red}{self.total_loss_pkt_num}{bcolors.end}'.ljust(19), f'{bcolors.bold}TD Schedule:{bcolors.end}'.ljust(13), f'{bcolors.purple}{self.TD_policy.name}{bcolors.end}')
            print(f'{bcolors.bold}Current ActiveUE:{bcolors.end}', f'{self.print_activate_ue}')

            # UE Information
            print(f'\n{bcolors.gray}-------------------------------------------------------------- UE -------------------------------------------------------------------{bcolors.end}')
            print(f'{bcolors.bold}UEid{bcolors.end}'.center(14), f'{bcolors.bold}Slice{bcolors.end}'.center(14), f'{bcolors.bold}Distance{bcolors.end}'.center(20), f'{bcolors.bold}SNR{bcolors.end}'.center(21), f'{bcolors.bold}Delay{bcolors.end}'.center(21),
             f'{bcolors.bold}Pdcprate{bcolors.end}'.center(27), f'{bcolors.bold}Buffer{bcolors.end}'.center(18), f'{bcolors.bold}Request{bcolors.end}'.center(11), f'{bcolors.bold}Receive{bcolors.end}'.center(19), f'{bcolors.bold}Loss_pkt{bcolors.end}'.center(5),
             f'{bcolors.bold}Sche_t{bcolors.end}'.center(17), f'{bcolors.bold}Avg_prb{bcolors.end}'.center(17))
            
            total_buffer,total_request,total_receive,total_loss,total_schedule,total_prb = 0,0,0,0,0,0
            for ue in self.UEs:

                # 只打印20个
                if ue.id > 20 and ue.id <= 23:
                    print('•'.center(5))
                    continue
                elif ue.id >= 24:
                    break

                print(f'{ue.id}'.center(5), f'{ue.slice}'.center(7), f'{bcolors.yellow}{ue.distance} m{bcolors.end}'.center(21), f'{round(ue.snr,2)}'.center(13), 
                f'{bcolors.blue}{round(ue.delay, 2)} ms{bcolors.end}'.center(23), f'{round(ue.pdcprate, 2)} b/ms'.center(17), f'{bcolors.purple}{len(ue.request_list)}{bcolors.end}'.center(20), 
                f'{ue.period_request_pkt_num}'.center(8),  f'{bcolors.green}{ue.period_receive_pkt_num}{bcolors.end}'.center(18), f'{bcolors.red}{ue.loss_pkt_num}{bcolors.end}'.center(18),
                f'{ue.period_schedule_time}'.center(9), f'{ue.period_avg_prb}'.center(10))

                # statistic
                total_buffer += len(ue.request_list)
                total_request += ue.period_request_pkt_num
                total_receive += ue.period_receive_pkt_num
                total_loss += ue.loss_pkt_num
                total_schedule += ue.period_schedule_time
                total_prb += ue.period_avg_prb

            print(f' total'.ljust(73), f'{bcolors.purple}{total_buffer}{bcolors.end}'.center(20), f'{total_request}'.center(8),  f'{bcolors.green}{total_receive}{bcolors.end}'.center(18),
             f'{bcolors.red}{total_loss}{bcolors.end}'.center(18), f'{total_schedule}'.center(9), f'{total_prb}'.center(10))

            print(f'{bcolors.gray}-------------------------------------------------------------------------------------------------------------------------------------{bcolors.end}\n')

        # 周期重置统计量
        self.reset_statistic()