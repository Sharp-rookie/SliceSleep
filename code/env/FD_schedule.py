# -*- coding: utf-8 -*-


class Scheme(object):

    def __init__(self, ueid, trans_rate):
        self.ueid = ueid               # ueid in BS side
        self.trans_rate = trans_rate   # transmitting rate
        self.packets = []              # selected packets
        self.requiredRB = 0            # required RB number
        self.reservedRB = 0            # reserved RB number


class RR_FD(object):
    """Round Robin algorithm for frequency domain schedule"""

    def __init__(self):
        self.name = 'RR'

    def checkUEBufferInfo(self, gnb, active_UE, result):
        """Calculate the required RB number of active UE"""

        if len(active_UE) == 0:
            return
        
        for ueid, ue_scheme in result.items():
            
            size = 0
            for pkt in ue_scheme.packets:
                size += pkt.size
            ue_scheme.requiredRB = size / (ue_scheme.trans_rate * (gnb.tti/1000))

    def schedule(self, gnb):
        """Allocate RB for UE to which selected packets belong""" 

        if len(gnb.scheduling_queue) == 0:
            return None

        # init the UE info to which selected packets belong
        result = dict()
        active_UE = []
        for pkt in gnb.scheduling_queue:
            if pkt.ueid not in active_UE:
                active_UE.append(pkt.ueid)
                result[pkt.ueid] = Scheme(pkt.ueid, gnb.UE[pkt.ueid].trans_rate)
            
            result[pkt.ueid].packets.append(pkt)
        
        # check UE required RB number
        self.checkUEBufferInfo(gnb, active_UE, result)

        # allocate prb
        totalPRB = gnb.total_PRB
        satisfied = False
        while totalPRB > 0 and not satisfied:
            satisfied = True
            for ueid in active_UE:
                ue_scheme = result[ueid]

                if totalPRB <= 0:
                    break
                
                if ue_scheme.reservedRB < ue_scheme.requiredRB:
                    satisfied = False
                    ue_scheme.reservedRB += 1
                    totalPRB -= 1
        
        # statistic
        gnb.prb_usage += gnb.total_PRB - totalPRB

        # clear scheduling queue, no matter whether the packet could be sent back
        gnb.scheduling_queue = []

        return result