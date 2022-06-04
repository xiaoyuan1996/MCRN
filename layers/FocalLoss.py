#encoding:utf-8
# -----------------------------------------------------------
# "MCRN: A Multi-source Cross-modal Retrieval Network for Remote Sensing"
# Zhiqiang Yuan, Changyuan Tian, Wenkai Zhang, Yongqiang Mao, Hongqi Wang, and Xian Sun.
# Undergoing review.
# Writen by YuanZhiqiang, 2022.  Our code is depended on AMFMN
# ------------------------------------------------------------

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.shape[-1]
    return ((1-epsilon) * inputs) + (epsilon / K)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, class_num=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.class_name = class_num

    def forward(self, inputs, targets):
#        target = []
#        for t in targets.cpu().numpy():
#            tmp = [0]*self.class_name
#            tmp[t] = 1
#            target.append(tmp)
#        targets = torch.Tensor(target).cuda()
        targets = label_smoothing(targets)

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    input_demo = np.array([0, 1, 0, 0])
    input_demo = Variable(torch.from_numpy(input_demo))
    print(input_demo)

    output_demo = label_smoothing(input_demo, epsilon=0.1)
    print(output_demo)
