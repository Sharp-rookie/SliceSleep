# -*- coding: utf-8 -*-


class Packet(object):
    """Request Packet"""

    def __init__(self, ueid, index1, index2, type, avg_size, time_slot):

        self.ueid = ueid                              # ueid in BS side
        self.slice = type                             # slice type
        self.id_UE_side = index1                      # index in UE side
        self.id_BS_side = index2                      # index in BS side
        self.prb = 0                                  # reserved PRB num
        self.startT = time_slot                       # start time slot, TTI
        self.delay = 0                                # delay = queue delay + transmitting delay, ms
        self.size = self.randomSize(type, avg_size)   # file size, bit
    
    def randomSize(self, type, avg_size):

        if type == 'eMBB':
            return int(8 * avg_size[0])  # bit = 8 * Byte
        elif type == 'URLLC':
            return int(8 * avg_size[1])
        elif type == 'mMTC':
            return int(8 * avg_size[2])