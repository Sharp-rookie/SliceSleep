# -*- coding: utf-8 -*-
import numpy as np


class Bucket():
    """
    The Token Bucket Shaper of severy data flows (Namely, a slice)
    """

    def __init__(self, config: list):

        # properties
        self.rate = config[0] # the rate of the bucket
        self.reservedPRB = config[1] # reserved PRB for this slice
        self.offset = config[2] # the offset of the bucket

    def reset(self, config: list):
        """
        Reset the bucket
        """

        self.rate = config[0]
        self.reservedPRB = config[1]
        self.offset = config[2]


class TokenBucket(object):
    """ Token Bucket Policy"""

    def __init__(self, bucket_config: list):
        self.name = 'TokenBucket'
        self.buckets =  [Bucket(config) for config in bucket_config]
        
        # 计时变量
        self.lastTTI = [0 for _ in range(len(self.buckets))]
        self.currentTTI = [0 for _ in range(len(self.buckets))]
        self.stay_time = [1 for _ in range(len(self.buckets))]

        # prb for [eMBB, URLLC, mMTC]
        self.reservedPRB = [self.buckets[i].reservedPRB for i in range(len(self.buckets))]
    
    def is_token_arrival(self, bucketID) -> bool:
        """
        Check if the token is arrived
        """
        
        # check sliceID
        assert(bucketID>=0 and bucketID<len(self.buckets))

        # update currentTTI
        self.currentTTI[bucketID] += 1

        # calculate stay time
        bucket = self.buckets[bucketID]
        stay_tti = int(np.ceil(bucket.rate * bucket.offset))

        # check stay time
        if self.stay_time[bucketID] < stay_tti:
            self.stay_time[bucketID] += 1
            return True
        
        # check token arrival
        if (self.currentTTI[bucketID] - self.lastTTI[bucketID]) % bucket.rate == 0:
            self.stay_time[bucketID] = 1
            self.lastTTI[bucketID] = self.currentTTI[bucketID]
            return True
        
        return False
    
    def schedule(self, gnb, prb_utilization) -> list:
        """
        Schedule the data flow
        """

        # check if the activeUEs list is full
        if len(gnb.activeUEs) == gnb.MAX_ACTIVE_UE:
            return [False]*3
            
        # check if the token is arrived
        eMBB = True if self.is_token_arrival(0) else False
        URLLC = True if self.is_token_arrival(1) else False
        mMTC = True if self.is_token_arrival(2) else False

        if not eMBB and not URLLC and not mMTC:
            return [True]*3

        # statistic
        # TODO: 这里应该统计整体的频带利用率，单个的无意义
        if eMBB:
            prb_utilization[1][0] += sum(self.reservedPRB)
        if URLLC:
            prb_utilization[1][1] += sum(self.reservedPRB)
        if mMTC:
            prb_utilization[1][2] += sum(self.reservedPRB)

        # select the activeUEs
        current_lastUE_id = gnb.lastUE_id
        while len(gnb.activeUEs)<gnb.MAX_ACTIVE_UE and len(gnb.activeUEs)<gnb.ueNum:
            ue = gnb.UEs[gnb.lastUE_id]
            if ue not in gnb.activeUEs:
                if ue.slice == 'eMBB' and eMBB:
                    gnb.activeUEs.append(ue)
                    ue.period_schedule_time += 1

                elif ue.slice == 'URLLC' and URLLC:
                    gnb.activeUEs.append(ue)
                    ue.period_schedule_time += 1
                    
                elif ue.slice == 'mMTC' and mMTC:
                    gnb.activeUEs.append(ue)
                    ue.period_schedule_time += 1
                
                else:
                    # print(ue.id, 'token missing')
                    pass
            
            gnb.lastUE_id = (gnb.lastUE_id+1)%gnb.ueNum

            if current_lastUE_id == gnb.lastUE_id: # 已经遍历了所有UE
                break

        return [not eMBB, not URLLC, not mMTC]

    def reset(self):
        self.lastTTI = [0 for _ in range(len(self.buckets))]
        self.currentTTI = [0 for _ in range(len(self.buckets))]
        self.stay_time = [1 for _ in range(len(self.buckets))]
