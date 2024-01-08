#encoding:utf-8
# -----------------------------------------------------------
# "MCRN: A Multi-source Cross-modal Retrieval Network for Remote Sensing"
# Zhiqiang Yuan, Changyuan Tian, Wenkai Zhang, Yongqiang Mao, Hongqi Wang, and Xian Sun.
# Undergoing review.
# Writen by YuanZhiqiang, 2022.  Our code is depended on AMFMN
# ------------------------------------------------------------

import argparse
import copy
import logging
import os

import numpy as np 
import torch
import yaml

import engine
import utils
from dataloader import data
from vocab import deserialize_vocab


def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSITMD_MCRN.yaml', type=str,
                        help='path to a yaml options file')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.safe_load(handle)

    return options

def main(options):
    # choose model
    if options['model']['name'] == "MCRN":
        from layers.MCRN import MCRN_Finetune as model
    elif options['model']['name'] == "VSE":
        from layers.VSE import VSE as model
    elif options['model']['name'] == "LWMCR":
        from layers.LW_MCM import LWMCR as model
    elif options['model']['name'] == "AMFMN":
        from layers.AMFMN import AMFMN as model
    elif options['model']['name'] == "SCAN":
        from layers.scan_model import SCAN_model as model
    else:
        raise NotImplementedError

    # make vocab
    vocab = deserialize_vocab(options['dataset']['vocab_path'])
    vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab_word = [tup[0] for tup in vocab_word]

    # Create dataset, model, criterion and optimizer
    test_loader = data.get_test_loader(vocab, options)

    # model
    model = model(options['model'], vocab_word).cuda()

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if os.path.isfile(options['optim']['resume']):
        print("=> loading checkpoint '{}'".format(options['optim']['resume']))
        checkpoint = torch.load(options['optim']['resume'])
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        model.load_state_dict(checkpoint['model'])
    else:
        print("=> no checkpoint found at '{}'".format(options['optim']['resume']))

    # evaluate on test set
    sims = engine.validate_test(test_loader, model)

    # ave
    [d_i2t, d_i2a, d_i2i] = sims

    # get indicators
    (r1it, r5it, r10it, medri, meanri), _ = utils.acc_i2t2(d_i2t)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1it, r5it, r10it, medri, meanri))
    (r1ti, r5ti, r10ti, medrt, meanrt), _ = utils.acc_t2i2(d_i2t)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1ti, r5ti, r10ti, medrt, meanrt))
    currscore_vt = (r1ti + r5ti + r10ti + r1it + r5it + r10it)/6.0

    all_score_1 = "I2T: r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1it, r5it, r10it, medri, meanri, r1ti, r5ti, r10ti, medrt, meanrt, currscore_vt
    )


    (r1ia, r5ia, r10ia, medri, meanri), _ = utils.acc_i2t2(d_i2a)
    logging.info("Image to audio: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1ia, r5ia, r10ia, medri, meanri))
    (r1ai, r5ai, r10ai, medrt, meanrt), _ = utils.acc_t2i2(d_i2a)
    logging.info("Audio to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1ai, r5ai, r10ai, medrt, meanrt))
    currscore_va = (r1ai + r5ai + r10ai + r1ia + r5ia + r10ia)/6.0

    all_score_2 = "I2A: r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1a:{} r5a:{} r10a:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1ia, r5ia, r10ia, medri, meanri, r1ai, r5ai, r10ai, medrt, meanrt, currscore_va
    )

    (r1ii, r5ii, r10ii, medri, meanri), _ = utils.acc_i2i(d_i2i)
    logging.info("Image to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1ii, r5ii, r10ii, medri, meanri))
    (r1ii2, r5ii2, r10ii2, medrt, meanrt), _ = utils.acc_i2i(np.transpose(d_i2i))
    logging.info("Image to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1ii2, r5ii2, r10ii2, medrt, meanrt))

    currscore_vv = (r1ii + r5ii + r10ii + r1ii2 + r5ii2 + r10ii2)/6.0

    all_score_3 = "I2I: r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1ii, r5ii, r10ii, medri, meanri, r1ii2, r5ii2, r10ii2, medrt, meanrt, currscore_vv
    )

    result = [r1it, r5it, r10it, r1ti, r5ti, r10ti, r1ia, r5ia, r10ia, r1ai, r5ai, r10ai, (r1ii2 + r1ii) * 0.5, (r5ii2 + r5ii) * 0.5, (r10ii2 + r10ii) * 0.5]
    result = result + [np.mean(result)]

    return result

def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['optim']['resume'] = options['logs']['ckpt_save_path'] + options['k_fold']['experiment_name'] + "/" \
                                         + str(k) + "/" + options['model']['name'] + '_best.pth.tar'

    return updated_options

if __name__ == '__main__':
    options = parser_options()

    # calc ave k results
    last_score = []
    for k in range(options['k_fold']['nums']):
        print("=========================================")
        print("Start evaluate {}th fold".format(k))

        # update save path
        update_options = update_options_savepath(options, k)

        # run experiment
        one_score = main(update_options)
        last_score.append(one_score)

        print("Complete evaluate {}th fold".format(k))

    # ave
    print("\n===================== Ave Score ({}-fold verify) =================".format(options['k_fold']['nums']))
    last_score = np.average(last_score, axis=0)
    names = 'r1it, r5it, r10it, r1ti, r5ti, r10ti, r1ia, r5ia, r10ia, r1ai, r5ai, r10ai, r1ii, r5ii, r10ii, mr'.split(", ")
    for idx, (name,score) in enumerate(zip(names,last_score)):
        if idx % 6 == 0:
            print("------------")
        print("{}:{}".format(name, score))

    print("\n==================================================================")

