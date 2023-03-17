# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import expon

from .packet import Packet


class UE(object):
    """User Equipment"""

    def __init__(self, id, type, tti, avg_interval, avg_size):
        
        # basic parameters
        self.id = id                                # ueid in BS side
        self.slice = type                           # slice type
        self.tti = tti                              # TTI, ms
        self.snr = None                             # channel signal-noise-ratio
        self.trans_rate = 0                         # transmitting rate, bit/s
        self.distance = np.random.randint(1, 161)   # distance to BS, m
        self.request_list = []                      # request list
        self.received = []                          # received request id in UE side
        self.removeCount = 0                        # record move tti

        # traffic parameters
        self.avg_size = avg_size   # mean file size, B
        self.interval = 100        # arrival interval, TTI
        self.duration = 0          # last arrival duration, TTI
        if self.slice == 'eMBB':   # request arrival interval distribution, TTI
            self.interval_distribution = expon(loc=0, scale=avg_interval[0]/self.tti)
        elif self.slice == 'URLLC':
            self.interval_distribution = expon(loc=0, scale=avg_interval[1]/self.tti)
        elif self.slice == 'mMTC':
            self.interval_distribution = expon(loc=0, scale=2*avg_interval[2]/self.tti)

        # statistic metrics
        self.delay = None                 # delay
        self.throughput = 0               # throughput
        self.loss_pkt_num = 0             # total number of lost packets
        self.total_request_pkt_num = 0    # total number of requests
        self.total_receive_pkt_num = 0    # total number of received packets
        self.period_request_pkt_num = 0   # number of requests per period
        self.period_receive_pkt_num = 0   # number of received packets per period
        
    def gen_request(self, BS_side_id, time_slot):
        """Generate a request"""

        # check if arrival
        if self.duration < self.interval:
            self.duration += 1
            return None

        self.duration = 0 # reset duration
                
        # update next arrival interval
        self.interval = np.ceil(self.interval_distribution.rvs())

        # id in UE side
        UE_side_id = self.request_list[-1].id_UE_side + 1 if len(self.request_list) else 0
        
        # statistic
        self.total_request_pkt_num += 1
        self.period_request_pkt_num += 1

        return Packet(self.id, UE_side_id, BS_side_id, self.slice, self.avg_size, time_slot)

    def randomMove(self):
        """Random move"""

        # move per second
        self.removeCount = (self.removeCount + 1) % round(1*1000/self.tti)
        if self.removeCount != 0:
            return

        # move
        if self.slice == 'eMBB':
            self.distance += np.random.randint(-1, 2)  # 1 m/s
        elif self.slice == 'URLLC':
            self.distance += np.random.randint(-5, 6)  # 5 m/s
        elif self.slice == 'mMTC':
            pass  # 0 m/s
        
        # distance in (0, 160) m
        self.distance = min(self.distance, 160)
        self.distance = max(self.distance, 0)
    
    def loss_pkt(self, packet):
        """loss packet cause BS buffer overflow"""

        self.loss_pkt_num += 1
        del packet
    
    def receive_pkt(self, packet: Packet):
        """Receive a send-back from BS"""
        
        # moving average delay
        self.delay = 0.5*self.delay + 0.5*packet.delay if self.delay is not None else packet.delay

        # record request id in UE side
        self.received.append(packet)

        # statistic
        self.throughput += packet.size
        self.total_receive_pkt_num += 1
        self.period_receive_pkt_num += 1
    
    def clear_received_request(self):
        """pop the received request from pkt_buffer"""

        for pkt in self.received:
            self.request_list.remove(pkt)
        self.received = []
    
    def reset_statistic_info(self):
        """reset statistic metrics per period"""

        self.delay = None
        self.throughput = 0
        self.loss_pkt_num = 0
        self.period_receive_pkt_num = 0
        self.period_request_pkt_num = 0
    
    def reset(self):
        """reset overall infomation"""

        # clear buffer
        for pkt in self.request_list:
            del pkt
        self.request_list = []
        self.received = []

        # reset overall statistic infomation
        self.reset_statistic_info()
        self.total_request_pkt_num = 0
        self.total_receive_pkt_num = 0

        # reset other vars
        self.interval = 100
        self.removeCount = 0
