#encoding:utf-8
# -----------------------------------------------------------
# "MCRN: A Multi-source Cross-modal Retrieval Network for Remote Sensing"
# Zhiqiang Yuan, Changyuan Tian, Wenkai Zhang, Yongqiang Mao, Hongqi Wang, and Xian Sun.
# Undergoing review.
# Writen by YuanZhiqiang, 2022.  Our code is depended on AMFMN
# ------------------------------------------------------------

# A revision version from Skip-thoughs
import skipthoughts


def factory(vocab_words, opt , dropout=0.25):
    if opt['arch'] == 'skipthoughts':
        st_class = getattr(skipthoughts, opt['type'])
        seq2vec = st_class(opt['dir_st'],
                           vocab_words,
                           dropout=dropout,
                           fixed_emb=opt['fixed_emb'])

    else:
        raise NotImplementedError
    return seq2vec
