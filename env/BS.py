# -*- coding: utf-8 -*-
import numpy as np

from .FD_schedule import RR_FD
from .TD_schedule import TokenBucket
from .ue import UE


def dBm2W(dBm):
    """Convert dBm to W"""
    return 10**(dBm/10) / 1000


class BaseStation(object):
    """Base station"""

    def __init__(self,
        ue_number: list,
        avg_interval: list,
        avg_size: list,
        tti: float,
        bucket_config: list
        ):

        # basic parameters
        self.time_slot = 0                       # system time slot, TTI
        self.tti = tti                           # TTI, ms
        self.duration = 0                        # simulation duration after last reset, TTI
        self.pkt_buffer = []                     # buffer
        self.buffer_size = 3000                  # buffer size
        self.total_PRB = 100                     # number of bandwidth units
        self.band_PRB = 1.8e5                    # B, hz
        self.frequency = 2000                    # carirrer frequency, Mhz
        self.trans_p = 46                        # transmitting power, dBm
        self.AWGN = -174                         # additional Gaussian white noise, dBm/hz
        self.fixed_power_wake = 266.8            # fixed power (wake-up mode), W
        self.fixed_power_sleep = 10.             # fixed power (deep-sleep mode), W
        self.load_power = 40.43                  # load-related power consumption per slice,W
        self.switch_power = 0.                   # mode-switching power consumption per slice, J/time

        # traffic parameters
        self.slices = ['eMBB', 'URLLC', 'mMTC']  # slice type
        self.UE = []                             # UE list
        self.slice_ueNum = ue_number             # UE num per Slice
        self.ueNum = 0                           # total UE number
        self.avg_size = avg_size                 # mean required file size, B
        self.avg_interval = avg_interval         # mean request arrival interval, s

        # statistic metrics
        self.total_request_pkt_num = 0           # total number of requests
        self.total_send_pkt_num = 0              # total number of sending-backs
        self.total_loss_pkt_num = 0              # total number of lost packets
        self.period_slice_request = [0, 0, 0]    # number of requests per period
        self.period_slice_send = [0, 0, 0]       # number of sending-backs per period
        self.period_slice_loss = [0, 0, 0]       # number of lost packets per period
        self.delay = [None, None, None]          # delay per slice, ms
        self.datavolume = [0, 0, 0]              # data volume per slice, bit
        self.throughput = [0, 0, 0]              # thoughput per slice, bit
        self.fixed_consumption = 0               # fixed power consumption, J
        self.load_consumption = [0, 0, 0]        # load-related power consumption per slice, J
        self.switch_consumption = [0, 0, 0]      # mode-switching power consumption per slice, J
        self.prb_usage = 0                       # PRB usage
        self.prb_utilization = 0                 # PRB utilization rate

        # time domain scheduling strategy
        self.TD_policy = TokenBucket(bucket_config)
        self.sleep = [False] * 3      # sleep marker of current TTI
        self.last_sleep = [False] * 3      # sleep marker of last TTI
        self.scheduling_num = 1000    # max number of scheduled request per TTI
        self.scheduling_queue = []    # list of the selected UE to be scheduled per TTI
        
        # frequency domain scheduling strategy
        self.FD_policy = RR_FD()

        self.init_UE(self.slice_ueNum)

    def init_UE(self, number: list):
        """Initialize UE of each slice"""

        for i in range(number[0]):
            self.UE.append(UE(i, 'eMBB', self.tti, self.avg_interval, self.avg_size))
        for i in range(number[1]):
            self.UE.append(UE(i+number[0], 'URLLC', self.tti, self.avg_interval, self.avg_size))
        for i in range(number[2]):
            self.UE.append(UE(i+number[0]+number[1], 'mMTC', self.tti, self.avg_interval, self.avg_size))
        
        self.ueNum = len(self.UE)

    def simulate(self):
        """A simulation pipeline per TTI"""

        self.time_slot += 1                     # update time slot
        self.duration += 1                      # update simulation duration

        self.query_ue_request()                 # query if UE request arrived
        self.update_ue_channel_condition()      # update UE channel condition
        self.sleep = self.select_request()      # time domain scheduling
        schedule_scheme = self.allocate_prb()   # frequency domain scheduling
        self.send_packet(schedule_scheme)       # send packet back
        self.power_consumption()                # calculate power consumption of this TTI
    
    def query_ue_request(self):
        """Iterate over all users and query requests"""        
        
        for ue in self.UE:
            
            # id in BS side
            BS_side_id = self.pkt_buffer[-1].id_BS_side + 1 if len(self.pkt_buffer) else 0
            
            if pkt := ue.gen_request(BS_side_id, self.time_slot):
                if len(self.pkt_buffer) >= self.buffer_size:
                    ue.loss_pkt(pkt)
                    self.total_loss_pkt_num += 1
                    self.period_slice_loss[self.slices.index(ue.slice)] += 1
                else:
                    ue.request_list.append(pkt)
                    self.pkt_buffer.append(pkt)
                
                # statistic
                self.total_request_pkt_num += 1
                self.datavolume[self.slices.index(ue.slice)] += pkt.size
                self.period_slice_request[self.slices.index(ue.slice)] += 1
    
    def update_ue_channel_condition(self):
        """Update UE channel condition and calculate transmitting rate"""

        for ue in self.UE:

            ue.randomMove() # UE random move

            # path loss
            if ue.distance > 10:
                Lbf = 32.5 + 20*np.log10(ue.distance) + 20*np.log10(self.frequency)
            else:
                Lbf = 52.5 + 20*np.log10(self.frequency)
            
            # SNR
            ue.snr = dBm2W(self.trans_p - Lbf) / (dBm2W(self.AWGN))

            # transmitting rate
            ue.trans_rate = self.band_PRB * np.log2(1 + ue.snr)
    
    def select_request(self):
        """Select request from buffer to be scheduled in this TTI"""
        return self.TD_policy.schedule(self)

    def allocate_prb(self):
        """Allocate RB for selected request"""
        return self.FD_policy.schedule(self)
    
    def send_packet(self, schedule_scheme):

        if not schedule_scheme:
            return

        for ueid, ue_scheme in schedule_scheme.items():
            ue = self.UE[ueid]
            sliceid = self.slices.index(ue.slice)
            available_throughput = ue_scheme.reservedRB * ue.trans_rate * (self.tti/1000)

            for pkt in ue_scheme.packets:
                if available_throughput <= 0:
                    break

                if pkt.size <= available_throughput:

                    available_throughput -= pkt.size

                    # calculate delay
                    queue_delay = (self.time_slot - pkt.startT) * self.tti # queuing delay
                    transmit_delay = pkt.size / (ue_scheme.reservedRB * ue.trans_rate + 1e-7) * 1000 # 发送时延，单位ms
                    pkt.delay = queue_delay + transmit_delay
                
                    # remove the packet from the BS buffer
                    for gnb_pkt in self.pkt_buffer:
                        if gnb_pkt.id_BS_side == pkt.id_BS_side:
                            self.pkt_buffer.remove(gnb_pkt)
                            break
                    
                    # UE receive packet
                    ue.receive_pkt(pkt)

                    # statistic
                    self.total_send_pkt_num += 1
                    self.period_slice_send[sliceid] += 1
                    self.throughput[sliceid] += pkt.size

                    ue.wait = False  # ensure UE receiced packet in this TTI

                    del pkt  # release cache
                
                else:
                    # send part of whole packet
                    pkt.size -= available_throughput
                    
                    # statistic
                    self.throughput[sliceid] += available_throughput

                    available_throughput = 0
            
            ue.clear_received_request() # UE remove the received packets

    def power_consumption(self):
        
        # fixed power consumption
        if all(self.sleep):  # deep sleep
            self.fixed_consumption += self.fixed_power_sleep * self.tti/1000
        else:
            self.fixed_consumption += self.fixed_power_wake * self.tti/1000
        
        for sliceid in range(len(self.slices)):
            # load-related power consumption
            if not self.sleep[sliceid]:
                self.load_consumption[sliceid] += self.load_power * self.tti/1000  # TODO: 负载量未定义
            
            # mode-switching power consumption
            self.switch_consumption[sliceid] += int(self.last_sleep[sliceid] ^ self.sleep[sliceid]) * self.switch_power

            self.last_sleep[sliceid] = self.sleep[sliceid]


    def statistic(self):
        
        # delay
        tmp_delay = [[] for _ in range(3)]
        for ue in self.UE:
            if ue.wait: # check if recevived packet in this TTI
                continue
            ue.wait = True # reset
            tmp_delay[self.slices.index(ue.slice)].append(ue.delay)
        for slice_id in range(3):
            self.delay[slice_id] = np.mean(tmp_delay[slice_id])
        
        # prb utilization
        self.prb_utilization = self.prb_usage / (self.total_PRB * self.duration)

    def reset_statistic_info(self):
        """Reset statistic metrics per period"""

        # UE
        for ue in self.UE:
            ue.reset_statistic_info()

        # BS
        self.period_slice_request = [0, 0, 0]
        self.period_slice_send = [0, 0, 0]
        self.period_slice_loss = [0, 0, 0]
        self.delay = [None, None, None]
        self.throughput = [0, 0, 0]
        self.datavolume = [0, 0, 0]
        self.fixed_consumption = 0
        self.load_consumption = [0, 0, 0]
        self.switch_consumption = [0, 0, 0]
        self.prb_usage = 0
        self.prb_utilization = 0

    def reset(self):
        """Reset overall infomation"""

        self.reset_statistic_info()
        
        # reset basic parameters
        self.duration = 0

        # clear buffer and scheduling queue
        self.pkt_buffer = []
        self.scheduling_queue = []

        # statistic infomation
        self.total_request_pkt_num = 0
        self.total_send_pkt_num = 0
        self.total_loss_pkt_num = 0
    
    def close(self):
        """Close the BS"""

        # delete data
        for ue in self.UE:
            for pkt in ue.request_list:
                del pkt
            del ue