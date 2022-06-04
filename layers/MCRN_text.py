#encoding:utf-8
# -----------------------------------------------------------
# "MCRN: A Multi-source Cross-modal Retrieval Network for Remote Sensing"
# Zhiqiang Yuan, Changyuan Tian, Wenkai Zhang, Yongqiang Mao, Hongqi Wang, and Xian Sun.
# Undergoing review.
# Writen by YuanZhiqiang, 2022.  Our code is depended on AMFMN
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import skipthoughts

def lstm_factory(vocab_words, opt , dropout=0.25):
    if opt['arch'] == 'skipthoughts':
        st_class = getattr(skipthoughts, opt['type'])
        seq2vec = st_class(opt['dir_st'],
                           vocab_words,
                           dropout=dropout,
                           fixed_emb=opt['fixed_emb'])

    else:
        raise NotImplementedError
    return seq2vec


class MCRN_text(nn.Module):
    def __init__(self, vocab, opt, out_dropout=-1):
        super(MCRN_text, self).__init__()
        self.opt = opt
        self.vocab_words = vocab

        self.seq2vec = lstm_factory(self.vocab_words, self.opt['seq2vec'], self.opt['seq2vec']['dropout'])

        self.to_out = nn.Linear(in_features=2400, out_features=self.opt['embed']['embed_dim'])
        self.dropout = out_dropout

    def forward(self, input_text ):
        x_t_vec = self.seq2vec(input_text)
        out = F.relu(self.to_out(x_t_vec))
        if self.dropout >= 0:
            out = F.dropout(out, self.dropout)

        return out

if __name__ == "__main__":
    from torch.autograd import Variable
    input_demo = Variable(torch.ones((10, 25)).long())

    import yaml
    # load model options
    with open("../option/RSITMD_AMFMN.yaml", 'r') as handle:
        options = yaml.load(handle)

    # make vocab
    vocab_word = ["word", "s"]

    model = MCRN_text(vocab= vocab_word, opt = options['model'])

    results = model(input_demo)

    print(model)
    print(results.shape)