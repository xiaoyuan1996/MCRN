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

try:
    from layers import MCRN_audio
    from layers import MCRN_text
    from layers import MCRN_visual
    from layers import MCRN_share
except:
    import sys
    sys.path.append("..")
    from layers import MCRN_audio
    from layers import MCRN_text
    from layers import MCRN_visual
    from layers import MCRN_share

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im, dim=-1)
    s = l2norm(s, dim=-1)
    w12 = im.mm(s.t())
    return w12

class MCRN_Pretrain(nn.Module):
    def __init__(self, opt={}, vocab_word=None):
        super(MCRN_Pretrain, self).__init__()

        # visual feature
        self.visual = MCRN_visual.MCRN_visual(opt = opt)

        # text feature
        self.text = MCRN_text.MCRN_text(vocab=vocab_word, opt=opt)

        # audio feature
        self.audio = MCRN_audio.MCRN_audio(opt = opt)

        # share layer
        self.share = MCRN_share.MCRN_SPTM(opt = opt)

        self.Eiters = 0


    def forward(self, img=None, text=None, audio=None):
        # visual self-supervised
        if img != None:
            img1, img2 = img

            # encoding anchor1
            fv1 = self.visual(img1)
            gv1 = self.share(fv1)

            # encoding anchor2
            fv2 = self.visual(img2)
            gv2 = self.share(fv2)

            return [fv1, fv2], [gv1, gv2]

        # text self-supervised
        elif text != None:
            text1, text2 = text

            # encoding anchor1
            fv1 = self.text(text1)
            gv1 = self.share(fv1)

            # encoding anchor2
            fv2 = self.text(text2)
            gv2 = self.share(fv2)

            return [fv1, fv2], [gv1, gv2]

        # audio self-supervised
        elif audio != None:
            audio1, audio2 = audio

            # encoding anchor1
            fv1 = self.visual(audio1)
            gv1 = self.share(fv1)

            # encoding anchor2
            fv2 = self.visual(audio2)
            gv2 = self.share(fv2)

            return [fv1, fv2], [gv1, gv2]

        else:
            return None


class MCRN_Finetune(nn.Module):
    def __init__(self, opt={}, vocab_word=None):
        super(MCRN_Finetune, self).__init__()

        # visual feature
        self.visual = MCRN_visual.MCRN_visual(opt = opt)

        # text feature
        self.text = MCRN_text.MCRN_text(vocab=vocab_word, opt=opt)

        # audio feature
        self.audio = MCRN_audio.MCRN_audio(opt = opt)

        # share layer
        self.share = MCRN_share.MCRN_SPTM(opt = opt)

        self.dim = opt['embed']['embed_dim']

        self.dropout = opt['embed']['share_dropout']

        self.Eiters = 0



    def forward(self, img, text, audio, text_lens=None):
        # batch
        batch_v = img.size(0)
        batch_t = text.size(0)
        batch_a = audio.size(0)

        fv = self.visual(img)
        ft = self.text(text)
        fa = self.audio(audio)

        fv = torch.dropout(fv, p=self.dropout, train=True)
        ft = torch.dropout(ft, p=self.dropout, train=True)
        fa = torch.dropout(fa, p=self.dropout, train=True)

        gv = self.share(fv)
        gt = self.share(ft)
        ga = self.share(fa)
        #
        # gv = fv
        # gt = ft
        # ga = fa

        # cosine
        score_vt = cosine_sim(gv, gt)
        score_va = cosine_sim(gv, ga)

        return [score_vt, score_va], [gv, gt, ga]

    def i2t(self, img, text):
        gv = self.share(self.visual(img))
        gt = self.share(self.text(text))
        return cosine_sim(gv, gt)

    def i2i(self, img, img2):
        gv1 = self.share(self.visual(img))
        gv2 = self.share(self.visual(img2))
        return cosine_sim(gv1, gv2)

    def i2a(self, img, audio):
        ga = self.share(self.audio(audio))
        gv = self.share(self.visual(img))
        return cosine_sim(gv, ga)

    def single_modal_encode(self, img):
        # encoding
        fv1 = self.visual(img)
        gv = self.share(fv1)

        return gv

if __name__ == "__main__":
    from torch.autograd import Variable
    input_demo = Variable(torch.zeros((10, 3, 256, 256)))
    text_demo = Variable(torch.ones((11, 25)).long())
    audio_demo = Variable(torch.zeros((12, 3, 256, 256)))

    import yaml
    # load model options
    with open("../option/RSITMD_MCRN.yaml", 'r') as handle:
        options = yaml.load(handle)


    vocab_word = ["word", "s"]

    model = MCRN_Finetune(options['model'], vocab_word)

    model(input_demo, text_demo, audio_demo)

    print(fv1.shape)
    print(gv1.shape)
    print(fv2.shape)
    print(gv2.shape)

    gv_3d = gv1[:,None,None,:].expand(10, 11, 12, 512)

    print(gv_3d)
    print(gv_3d.shape)


