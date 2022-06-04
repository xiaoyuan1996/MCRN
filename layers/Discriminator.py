#encoding:utf-8
# -----------------------------------------------------------
# "MCRN: A Multi-source Cross-modal Retrieval Network for Remote Sensing"
# Zhiqiang Yuan, Changyuan Tian, Wenkai Zhang, Yongqiang Mao, Hongqi Wang, and Xian Sun.
# Undergoing review.
# Writen by YuanZhiqiang, 2022.  Our code is depended on AMFMN
# ------------------------------------------------------------

import torch.nn as nn
import torch.nn.init

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        in_size = opt['discriminator']['in_size']
        mid_size = opt['discriminator']['mid_size']
        out_size = opt['discriminator']['out_size']
        dropout_r = opt['discriminator']['dropout_r']

        self.fc1 = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=True)

        self.linear = nn.Linear(mid_size, out_size)

        self.Eiters = 0


    def forward(self, x):
        out = self.fc1(x)
        return self.linear(out)


if __name__ == "__main__":
    import numpy as np
    from torch.autograd import Variable

    input_demo = Variable(torch.zeros((10, 512)))

    opt = {"discriminator":{
        "in_size":512,
        "mid_size":128,
        "out_size":3,
        "dropout_r":0.1
    }}

    model = Discriminator(opt)

    out = model(input_demo)
    print(out.shape)


    discriminator_loss = torch.nn.CrossEntropyLoss()

    batch_size = out.size(0)

    lable = torch.zeros(batch_size).long() + 1

    pred_y = torch.max(out, 1)[1].data.cpu().numpy()
    label = lable.data.numpy()

    accuracy = sum(np.equal(pred_y, label))
    print(accuracy)

    # loss = discriminator_loss(out, lable)
    # print(loss)

