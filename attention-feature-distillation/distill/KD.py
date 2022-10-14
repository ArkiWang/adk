from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=1):
        super(DistillKL, self).__init__()
        self.T = T
        # print(self.T.split('_'))
        # self.stu_Temp, self.tea_Temp = [float(T) for T in self.T.split('_')]

    def forward(self, y_s, y_t):
        tea_std = torch.std(y_t, dim=-1,keepdim=True)
        stu_std= torch.std(y_s, dim=-1, keepdim=True)

        self.tea_Temp = tea_std
        self.stu_Temp = stu_std

        p_s = F.log_softmax(y_s/self.stu_Temp, dim=1)
        p_t = F.softmax(y_t/self.tea_Temp, dim=1)
      
        loss = torch.sum(torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=-1) * (9 * torch.ones(y_s.shape[0],1).cuda())) /y_s.shape[0]/ y_s.shape[0]
        return loss