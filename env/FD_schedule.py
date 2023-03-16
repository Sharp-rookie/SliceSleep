# -*- coding: utf-8 -*-


slice = ['eMBB', 'URLLC', 'mMTC']

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
