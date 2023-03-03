# -*- coding: utf-8 -*-
__all__ = [
    'RR_slice_FD', 
    'RR_FD', 
    'PF_FD',
    ]

slice = ['eMBB', 'URLLC', 'mMTC']


class RR_slice_FD(object):
    """频域 Slice-based Round Robin 调度算法"""

    def __init__(self):

        self.name = 'RR_slice'

    def schedule(self, reservedPRB: list, activeUEs: list, prb_statistic: list):
        """逐业务做RR mode调度

        Parameters:
        ------- 
            reservedPRB: 每个业务的预留PRB个数
            activateUEs: 活跃用户列表
            prb_statistic: PRB统计信息
        """ 

        if len(activeUEs) == 0:
            return

        # 统计可用PRB数量
        if 'eMBB' in [ue.slice for ue in activeUEs]:
            prb_statistic[1][0] += reservedPRB[0]
        if 'URLLC' in [ue.slice for ue in activeUEs]:
            prb_statistic[1][1] += reservedPRB[1]
        if 'mMTC' in [ue.slice for ue in activeUEs]:
            prb_statistic[1][2] += reservedPRB[2]

        # allot prb
        sacrified = False
        embbPRB, urllcPRB, mmtcPRB = reservedPRB
        while (embbPRB != 0 or urllcPRB != 0 or mmtcPRB != 0) and not sacrified:
            sacrified = True
            for ue in activeUEs:
                if embbPRB <= 0 and urllcPRB <= 0 and mmtcPRB <= 0:
                    break
                
                if ue.slice=='eMBB' and embbPRB>0 and ue.prb-ue.requestPRB<=0:
                    sacrified = False
                    ue.prb += 1
                    embbPRB -= 1
                    prb_statistic[0][0] += 1 # 统计实际使用PRB数量
                    ue.period_avg_prb += 1
                elif ue.slice=='URLLC' and urllcPRB>0 and ue.prb-ue.requestPRB<=0:
                    sacrified = False
                    ue.prb += 1
                    urllcPRB -= 1
                    prb_statistic[0][1] += 1
                    ue.period_avg_prb += 1
                elif ue.slice=='mMTC' and mmtcPRB>0 and ue.prb-ue.requestPRB<=0:
                    sacrified = False
                    ue.prb += 1
                    mmtcPRB -= 1
                    prb_statistic[0][2] += 1
                    ue.period_avg_prb += 1
                else:
                    continue
                            

class RR_FD(object):
    """频域 Round Robin 调度算法"""

    def __init__(self):

        self.name = 'RR'

    def schedule(self, reservedPRB: list, activeUEs: list, prb_statistic: list):
        """不区分业务，统一做RR mode调度

        Parameters:
        ------- 
            reservedPRB: 每个业务的预留PRB个数
            activateUEs: 活跃用户列表
            prb_statistic: PRB统计信息
        """ 

        if len(activeUEs) == 0:
            return

        totalPRB = sum(reservedPRB)

        # 统计可用PRB数量
        if 'eMBB' in [ue.slice for ue in activeUEs]:
            prb_statistic[1][0] += totalPRB
        if 'URLLC' in [ue.slice for ue in activeUEs]:
            prb_statistic[1][1] += totalPRB
        if 'mMTC' in [ue.slice for ue in activeUEs]:
            prb_statistic[1][2] += totalPRB

        # allot prb
        sacrified = False
        while totalPRB != 0 and not sacrified:
            sacrified = True
            for ue in activeUEs:
                sliceid = slice.index(ue.slice)
                if totalPRB <= 0:
                    break
                if ue.prb-ue.requestPRB <= 0:
                    sacrified = False
                    ue.prb += 1
                    totalPRB -= 1
                    prb_statistic[0][sliceid] += 1
                    ue.period_avg_prb += 1


# TODO: 需要验证目前定义的优先级的合理性，通过特定场景来验证
class PF_FD(object):
    """频域 Proportional Fair 调度算法"""

    def __init__(self):

        self.name = 'PF'

    def schedule(self, reservedPRB: list, activeUEs: list, prb_statistic: list):
        """不区分业务，统一做Proportion Fair调度

        Parameters:
        ------- 
            reservedPRB: 每个业务的预留PRB个数
            activateUEs: 活跃用户列表
            prb_statistic: PRB统计信息
        """ 

        if len(activeUEs) == 0: # 无活跃用户则返回
            return

        totalPRB = sum(reservedPRB)

        # 统计可用PRB数量
        if 'eMBB' in [ue.slice for ue in activeUEs]:
            prb_statistic[1][0] += totalPRB
        if 'URLLC' in [ue.slice for ue in activeUEs]:
            prb_statistic[1][1] += totalPRB
        if 'mMTC' in [ue.slice for ue in activeUEs]:
            prb_statistic[1][2] += totalPRB

        # update UE priority
        for ue in activeUEs:
            ue.priority = ue.buffersize / (ue.pdcprate + 1e-3)

            # reset throughput
            ue.throughput = 0
        
        # sort UE by priority
        activeUEs.sort(key=lambda x: x.priority, reverse=True)

        # allot prb
        for ue in activeUEs:
            sliceid = slice.index(ue.slice)
            while totalPRB>0 and ue.prb-ue.requestPRB<0:
                ue.prb += 1
                totalPRB -= 1
                prb_statistic[0][sliceid] += 1
                ue.period_avg_prb += 1
