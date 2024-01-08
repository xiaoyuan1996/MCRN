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
import random

import torch
import yaml 
from torch.optim import *

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
    else:
        raise NotImplementedError

    # make ckpt save dir
    if not os.path.exists(options['logs']['ckpt_save_path']):
        os.makedirs(options['logs']['ckpt_save_path'])

    # make vocab
    vocab = deserialize_vocab(options['dataset']['vocab_path'])
    vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab_word = [tup[0] for tup in vocab_word]

    # model
    model = model(options['model'], vocab_word).cuda()

    # Create dataset, model, criterion and optimizer
    train_loader, val_loader = data.get_loaders(vocab, options)

    if options['model']['name'] == "MCRN":
        optimizer = torch.optim.Adam([
          {'params': filter(lambda p: p.requires_grad, model.share.parameters()), 'lr':   options['optim']['lr_finetune'] },
          {'params': filter(lambda p: p.requires_grad, model.visual.parameters()), 'lr':  0.3* options['optim']['lr_finetune'] },
          {'params': filter(lambda p: p.requires_grad, model.text.parameters()), 'lr':   0.3* options['optim']['lr_finetune'] },
          {'params': filter(lambda p: p.requires_grad, model.audio.parameters()), 'lr':  0.3* options['optim']['lr_finetune'] }
          ],
        )
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=options['optim']['lr_finetune'])

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if options['optim']['resume']:
        if os.path.isfile(options['optim']['resume']):
            print("=> loading checkpoint '{}'".format(options['optim']['resume']))
            checkpoint = torch.load(options['optim']['resume'])
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'], strict=False)

            # Eiters is used to show logs as the continuation of another
            model.Eiters = checkpoint['Eiters']

            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(options['optim']['resume'], start_epoch, best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(options['optim']['resume']))
    else:
        start_epoch = 0

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20)

    # Train the Model
    best_rsum = 0
    best_score = ""

    for epoch in range(0, options['optim']['epochs']):

        #utils.adjust_learning_rate(options, optimizer, epoch)

        scheduler.step()
        print("lr: {} and {}".format(optimizer.state_dict()['param_groups'][0]['lr'], optimizer.state_dict()['param_groups'][-1]['lr']))

        # train for one epoch
        engine.train(train_loader, model, optimizer, epoch, opt=options)

        # evaluate on validation set
        if epoch % options['logs']['eval_step'] == 0:
            rsum, all_scores = engine.validate(val_loader, model)

            is_best = rsum > best_rsum
            if is_best:
                best_score = all_scores
            best_rsum = max(rsum, best_rsum)

            # save ckpt
            utils.save_checkpoint(
                {
                'epoch': epoch + 1,
                'arch': 'baseline',
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'options': options,
                'Eiters': model.Eiters,
            },
                is_best,
                filename='ckpt_{}_{}_{:.2f}.pth.tar'.format(options['model']['name'] ,epoch, best_rsum),
                prefix=options['logs']['ckpt_save_path'],
                model_name=options['model']['name']
            )

            print("Current {}th fold.".format(options['k_fold']['current_num']))
            print("Now  score:")
            print(all_scores)
            print("Best score:")
            print(best_score)

            utils.log_to_txt(
                contexts= "Experiment:{}, Epoch:{} ".format(options['k_fold']['experiment_name'], epoch+1) + all_scores,
                filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
            )
            utils.log_to_txt(
                contexts= "Best:   " + best_score,
                filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
            )

def generate_random_samples(options):
    # load all anns
    audios = utils.load_from_txt(options['dataset']['data_path'] + 'train_audio.txt')
    caps = utils.load_from_txt(options['dataset']['data_path'] + 'train_caps.txt')
    fnames = utils.load_from_txt(options['dataset']['data_path'] + 'train_filename.txt')

    # merge
    assert len(caps) // 5 == len(fnames)
    all_infos = []
    for img_id in range(len(fnames)):
        cap_id = [img_id * 5, (img_id + 1) * 5]
        all_infos.append([audios[cap_id[0]:cap_id[1]] ,caps[cap_id[0]:cap_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    percent = 0.8
    train_infos = all_infos[:int(len(all_infos) * percent)]
    val_infos = all_infos[int(len(all_infos) * percent):]

    # save to txt
    train_audios = []
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for audio in item[0]:
            train_audios.append(audio)
        for cap in item[1]:
            train_caps.append(cap)
        train_fnames.append(item[2])
    utils.log_to_txt(train_audios, options['dataset']['data_path'] + 'train_audios_verify.txt', mode='w')
    utils.log_to_txt(train_caps, options['dataset']['data_path'] + 'train_caps_verify.txt', mode='w')
    utils.log_to_txt(train_fnames, options['dataset']['data_path'] + 'train_filename_verify.txt', mode='w')

    val_audios = []
    val_caps = []
    val_fnames = []
    for item in val_infos:
        for audio in item[0]:
            val_audios.append(audio)
        for cap in item[1]:
            val_caps.append(cap)
        val_fnames.append(item[2])
    utils.log_to_txt(val_audios, options['dataset']['data_path'] + 'val_audios_verify.txt', mode='w')
    utils.log_to_txt(val_caps, options['dataset']['data_path'] + 'val_caps_verify.txt', mode='w')
    utils.log_to_txt(val_fnames, options['dataset']['data_path'] + 'val_filename_verify.txt', mode='w')

    print("Generate random samples to {} complete.".format(options['dataset']['data_path']))

def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['k_fold']['current_num'] = k
    updated_options['logs']['ckpt_save_path'] = options['logs']['ckpt_save_path'] + \
                                                options['k_fold']['experiment_name'] + "/" + str(k) + "/"
    return updated_options

if __name__ == '__main__':
    options = parser_options()

    # make logger
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # k_fold verify
    for k in range(options['k_fold']['nums']):
        print("=========================================")
        print("Start {}th fold".format(k))

        # generate random train and val samples
        generate_random_samples(options)

        # update save path
        update_options = update_options_savepath(options, k)

        # run experiment
        main(update_options)
