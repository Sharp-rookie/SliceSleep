# -*- coding: utf-8 -*-
import numpy as np


class SleepController(object):
    """Slice sleep controller"""

    def __init__(self, config: list):

        self.period = config[0]          # the period of a scheduling cycle
        self.wakeup_ratio = config[1]    # the time duration ratio of the wake-up mode

    def reset(self, config: list):
        """Reset the bucket"""

        self.period = config[0]
        self.wakeup_ratio = config[1]


class TokenBucket(object):

    def __init__(self, bucket_config: list):
        
        self.buckets =  [SleepController(config) for config in bucket_config]
        
        # time counter
        self.lastTTI = [0 for _ in range(len(self.buckets))]
        self.currentTTI = [0 for _ in range(len(self.buckets))]
        self.stay_time = [1 for _ in range(len(self.buckets))]

    def check_token(self, bucketID) -> bool:
        """
        Check if the token is arrived
        """
        
        # check sliceID
        assert(bucketID>=0 and bucketID<len(self.buckets))

        # update currentTTI
        self.currentTTI[bucketID] += 1

        # calculate stay time
        bucket = self.buckets[bucketID]
        stay_tti = int(np.ceil(bucket.period * bucket.wakeup_ratio))

        # check stay time
        if self.stay_time[bucketID] < stay_tti:
            self.stay_time[bucketID] += 1
            return True  # wake-up
        
        # check token arrival
        if (self.currentTTI[bucketID] - self.lastTTI[bucketID]) % bucket.period == 0:
            self.stay_time[bucketID] = 1
            self.lastTTI[bucketID] = self.currentTTI[bucketID]
            return True  # wake-up
        
        return False  # sleep
    
    def schedule(self, gnb) -> list:
        """Select the UE for scheduling queue"""

        # check if the scheduling queue is full
        assert len(gnb.scheduling_queue) == 0, f"scheduling queue must clear after scheduling in each TTI"
        
        # check if the slice if wake-up(True)
        eMBB = True if self.check_token(0) else False
        URLLC = True if self.check_token(1) else False
        mMTC = True if self.check_token(2) else False

        if not eMBB and not URLLC and not mMTC:
            return [True]*3  # deep sleep mode
        
        # select the request to be scheduled in this TTI
        for pkt in gnb.pkt_buffer:
            
            if len(gnb.scheduling_queue) >= gnb.scheduling_num:
                break
            
            if pkt.slice == 'eMBB' and eMBB:
                gnb.scheduling_queue.append(pkt)

            elif pkt.slice == 'URLLC' and URLLC:
                gnb.scheduling_queue.append(pkt)
                
            elif pkt.slice == 'mMTC' and mMTC:
                gnb.scheduling_queue.append(pkt)

        return [not eMBB, not URLLC, not mMTC]  # sleep--True, wake-up--False