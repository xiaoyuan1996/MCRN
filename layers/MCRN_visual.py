#encoding:utf-8
# -----------------------------------------------------------
# "MCRN: A Multi-source Cross-modal Retrieval Network for Remote Sensing"
# Zhiqiang Yuan, Changyuan Tian, Wenkai Zhang, Yongqiang Mao, Hongqi Wang, and Xian Sun.
# Undergoing review.
# Writen by YuanZhiqiang, 2022.  Our code is depended on AMFMN
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init
from torchvision.models.resnet import resnet18

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class ExtractFeature(nn.Module):
    def __init__(self, opt = {}, finetune=True, obtain_features=True):
        super(ExtractFeature, self).__init__()

        self.embed_dim = opt['embed']['embed_dim']
        self.obtain_feature = obtain_features

        self.resnet = resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = finetune

        self.pool_2x2 = nn.MaxPool2d(4)

        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_sample_4 = nn.Upsample(scale_factor=4, mode='nearest')

        self.linear = nn.Linear(in_features=512, out_features=self.embed_dim)

    def forward(self, img):
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f1 = self.resnet.layer1(x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        # Lower Feature
        f2_up = self.up_sample_2(f2)
        lower_feature = torch.cat([f1, f2_up], dim=1)

        # Higher Feature
        f4_up = self.up_sample_2(f4)
        higher_feature = torch.cat([f3, f4_up], dim=1)
        # higher_feature = self.up_sample_4(higher_feature)

        # batch * 512
        feature = f4.view(f4.shape[0], 512, -1)
        solo_feature = self.linear(torch.mean(feature,dim=-1))

        # torch.Size([10, 192, 64, 64])
        # torch.Size([10, 768, 64, 64])
        # torch.Size([10, 512])
        if self.obtain_feature:
            return lower_feature, higher_feature, solo_feature
        else:
            return solo_feature

class VSA_Module(nn.Module):
    def __init__(self, opt = {}):
        super(VSA_Module, self).__init__()

        # extract value
        channel_size = opt['multiscale']['multiscale_input_channel']
        out_channels = opt['multiscale']['multiscale_output_channel']
        embed_dim = opt['embed']['embed_dim']

        # sub sample
        self.LF_conv = nn.Conv2d(in_channels=192, out_channels=channel_size, kernel_size=3, stride=4)
        self.HF_conv = nn.Conv2d(in_channels=768, out_channels=channel_size, kernel_size=1, stride=1)

        # visual attention
        self.conv1x1_1 = nn.Conv2d(in_channels=channel_size*2, out_channels=out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=channel_size*2, out_channels=out_channels, kernel_size=1)

        # solo attention
        self.solo_attention = nn.Linear(in_features=256, out_features=embed_dim)

    def forward(self, lower_feature, higher_feature, solo_feature):

        # b x channel_size x 16 x 16
        lower_feature = self.LF_conv(lower_feature)
        higher_feature = self.HF_conv(higher_feature)

        # concat
        concat_feature = torch.cat([lower_feature, higher_feature], dim=1)

        # residual
        concat_feature = higher_feature.mean(dim=1,keepdim=True).expand_as(concat_feature) + concat_feature

        # attention
        main_feature = self.conv1x1_1(concat_feature)
        attn_feature = torch.sigmoid(self.conv1x1_2(concat_feature).view(concat_feature.shape[0],1,-1)).view(concat_feature.shape[0], 1, main_feature.shape[2], main_feature.shape[3])
        atted_feature = (main_feature*attn_feature).squeeze(dim=1).view(attn_feature.shape[0], -1)

       # solo attention
        solo_att = torch.sigmoid(self.solo_attention(atted_feature))
        solo_feature = solo_feature*solo_att

        return solo_feature

class MCRN_visual(nn.Module):
    def __init__(self, opt={}):
        super(MCRN_visual, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature(opt = opt)

        # vsa feature
        self.mvsa =VSA_Module(opt = opt)

    def forward(self, img):
        # extract features
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues
        mvsa_feature = self.mvsa(lower_feature, higher_feature, solo_feature)

        return mvsa_feature


if __name__ == "__main__":
    from torch.autograd import Variable
    input_demo = Variable(torch.zeros((10, 3, 256, 256)))

    import yaml
    # load model options
    with open("../option/RSITMD_AMFMN.yaml", 'r') as handle:
        options = yaml.load(handle)

    model = MCRN_visual(options['model'])

    results = model(input_demo)

    print(model)
    print(results.shape)
