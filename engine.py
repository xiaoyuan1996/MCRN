#encoding:utf-8
# -----------------------------------------------------------
# "MCRN: A Multi-source Cross-modal Retrieval Network for Remote Sensing"
# Zhiqiang Yuan, Changyuan Tian, Wenkai Zhang, Yongqiang Mao, Hongqi Wang, and Xian Sun.
# Undergoing review.
# Writen by YuanZhiqiang, 2022.  Our code is depended on AMFMN
# ------------------------------------------------------------

import logging
import os.path
import pickle
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from tqdm import tqdm

import utils
from loss_func.knowledge_distillation import KDLoss


def pretrain(train_loader, model, discriminator, optimizer, optimizer_D, epoch, opt={}):

    # loss
    intra_loss = utils.intra_loss
    inter_loss = torch.nn.MSELoss()
    discriminator_loss = torch.nn.CrossEntropyLoss()

    # extract value
    print_freq = opt['logs']['print_freq']

    # switch to train mode
    model.train()
    discriminator.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())
    for i, train_data in enumerate(train_loader):
        image_1, image_2, audio_1, audio_2, caption_1, caption_2, lengths_1, lengths_2 = train_data

        batch_size = image_1.size(0)

        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual_1 = Variable(image_1)
        input_visual_2 = Variable(image_2)
        input_text_1 = Variable(caption_1)
        input_text_2 = Variable(caption_2)
        input_audio_1 = Variable(audio_1)
        input_audio_2 = Variable(audio_2)

        if torch.cuda.is_available():
            input_visual_1 = input_visual_1.cuda()
            input_visual_2 = input_visual_2.cuda()
            input_text_1 = input_text_1.cuda()
            input_text_2 = input_text_2.cuda()
            input_audio_1 = input_audio_1.cuda()
            input_audio_2 = input_audio_2.cuda()

        ######## optimizer ########


        datatype = np.random.choice(['image', 'audio', 'text'])
        # datatype = np.random.choice(['image'])
        if datatype == 'image':
            [fv1, fv2], [gv1, gv2] = model(img = [input_visual_1, input_visual_2])
            discriminator_lable = torch.zeros(batch_size).long().cuda()
            fake_lable = torch.zeros(batch_size).long().cuda() + np.random.choice([1, 2])
        elif datatype == 'text':
            [fv1, fv2], [gv1, gv2] = model(text = [input_text_1, input_text_2])
            discriminator_lable = torch.zeros(batch_size).long().cuda() + 1
            fake_lable = torch.zeros(batch_size).long().cuda() + np.random.choice([0, 2])
        else:
            [fv1, fv2], [gv1, gv2] = model(audio = [input_audio_1, input_audio_2])
            discriminator_lable = torch.zeros(batch_size).long().cuda() + 2
            fake_lable = torch.zeros(batch_size).long().cuda() + np.random.choice([0, 1])

        if i%1 == 0:
            optimizer_D.zero_grad()

            dis_out1 = discriminator(gv1.detach())
            dis_out2 = discriminator(gv2.detach())
            D_L = discriminator_loss(dis_out1, discriminator_lable) + discriminator_loss(dis_out2, discriminator_lable)

            D_L.backward()
            optimizer_D.step()

        if i%3 == 0:

            optimizer.zero_grad()

            # calc loss
            loss_intra = 0.01 * intra_loss(fv1, fv2)
            loss_inter = inter_loss(gv1, gv2)

            # norm loss
            norm_loss = utils.norm_loss(fv1) + utils.norm_loss(fv2) + utils.norm_loss(gv1) + utils.norm_loss(gv2)


            dis_out1 = discriminator(gv1)
            dis_out2 = discriminator(gv2)
            D_loss = discriminator_loss(dis_out1, fake_lable) + discriminator_loss(
                dis_out2, fake_lable)
            G_loss = loss_intra + loss_inter + D_loss + 0.01 * norm_loss

            G_loss.backward()

            optimizer.step()
        ###############################

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                    'Sche:{}/{},loss_intra:{}, loss_inter:{}, D_loss:{}, norm_loss:{}, D_L:{}, batch_time:{}'.format(i, len(train_loader), loss_intra, loss_inter, D_loss, norm_loss, D_L,  batch_time)
            )

            utils.log_to_txt(
                'Sche:{}/{},G_L:{}, D_L:{}, batch_time:{}'.format(i, len(train_loader),G_loss, D_loss, batch_time),
                opt['logs']['ckpt_save_path']+ opt['model']['name'] + "_" + opt['dataset']['datatype'] +"_pretrain.txt"
            )


def preval(val_loader, model, discriminator,  opt={}):

    # loss
    intra_loss = torch.nn.MSELoss()
    inter_loss = torch.nn.MSELoss()

    model.eval()
    discriminator.eval()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())

    acc_all, diff_fv_all, diff_gv_all = [], [], []

    for i, train_data in tqdm.tqdm(enumerate(val_loader), desc="start eval ..."):
        image_1, image_2, audio_1, audio_2, caption_1, caption_2, lengths_1, lengths_2 = train_data

        batch_size = image_1.size(0)

        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual_1 = Variable(image_1)
        input_visual_2 = Variable(image_2)
        input_text_1 = Variable(caption_1)
        input_text_2 = Variable(caption_2)
        input_audio_1 = Variable(audio_1)
        input_audio_2 = Variable(audio_2)

        if torch.cuda.is_available():
            input_visual_1 = input_visual_1.cuda()
            input_visual_2 = input_visual_2.cuda()
            input_text_1 = input_text_1.cuda()
            input_text_2 = input_text_2.cuda()
            input_audio_1 = input_audio_1.cuda()
            input_audio_2 = input_audio_2.cuda()

        ######## optimizer ########

        datatype = np.random.choice(['image', 'audio', 'text'])
        # datatype = np.random.choice(['image'])
        if datatype == 'image':
            [fv1, fv2], [gv1, gv2] = model(img = [input_visual_1, input_visual_2])
            discriminator_lable = torch.zeros(batch_size).long().cuda()
        elif datatype == 'text':
            [fv1, fv2], [gv1, gv2] = model(text = [input_text_1, input_text_2])
            discriminator_lable = torch.zeros(batch_size).long().cuda() + 1
        else:
            [fv1, fv2], [gv1, gv2] = model(audio = [input_audio_1, input_audio_2])
            discriminator_lable = torch.zeros(batch_size).long().cuda() + 2


        # acc
        dis_out1 = discriminator(gv1.detach())
        dis_out2 = discriminator(gv2.detach())

        pred_y1 = torch.max(dis_out1, 1)[1].data.cpu().numpy()
        pred_y2 = torch.max(dis_out2, 1)[1].data.cpu().numpy()
        label = discriminator_lable.cpu().data.numpy()
        acc_all.append( (sum(np.equal(pred_y1, label)) + sum(np.equal(pred_y2, label)))*1.0 / (2 * batch_size))
        # loss
        loss_intra = intra_loss(fv1, fv2).cpu().data.numpy()
        loss_inter = inter_loss(gv1, gv2).cpu().data.numpy()

        diff_fv_all.append(loss_intra)
        diff_gv_all.append(loss_inter)

    return np.mean(acc_all), np.mean(diff_fv_all), np.mean(diff_gv_all)



def train(train_loader, model, optimizer, epoch, opt={}):

    # loss
    single_modal_loss = torch.nn.MSELoss()
    multi_modal_loss_mse = torch.nn.MSELoss()
    multi_modal_loss = KDLoss(temperature=0.2)

    # extract value
    grad_clip = opt['optim']['grad_clip']
    margin = opt['optim']['margin']
    print_freq = opt['logs']['print_freq']

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())
    for i, train_data in enumerate(train_loader):
        image_1, image_2, audio, captions, lengths, ids= train_data

        batch_size = image_1.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual_1 = Variable(image_1)
        input_visual_2 = Variable(image_2)
        input_text = Variable(captions)
        input_audio = Variable(audio)

        if torch.cuda.is_available():
            input_visual_1 = input_visual_1.cuda()
            input_visual_2 = input_visual_2.cuda()
            input_text = input_text.cuda()
            input_audio = input_audio.cuda()


        scores, [gv, gt, ga] = model(input_visual_1, input_text, input_audio, text_lens=lengths)
        gv2 = model.single_modal_encode(input_visual_2)

        torch.cuda.synchronize()

        # loss calc

        if opt['optim']['methods'] == "tpt":
            loss = utils.multi_source_triplet_loss(scores, margin)
            train_logger.update('L', loss.cpu().data.numpy())
        else:

            # multi-source triple loss
            Lmt = utils.multi_source_triplet_loss(scores, margin)

            # single-modal alignment loss
            Ls = single_modal_loss(gv, gv2) * 1000

            # multi-modal alignment loss
            Lm = (multi_modal_loss(gv, gt) + multi_modal_loss(gv, ga) + multi_modal_loss(ga, gt)) * 1000
            Lm_mse = (multi_modal_loss_mse(gv, gt) + multi_modal_loss_mse(gv, ga) + multi_modal_loss_mse(ga, gt)) * 1000

            # loss = utils.calcul_loss(scores, input_visual.size(0), margin, max_violation=max_violation, )
            loss = Lmt + Ls + Lm + Lm_mse

            train_logger.update('L', loss.cpu().data.numpy())
            train_logger.update('Lmt', Lmt.cpu().data.numpy())
            train_logger.update('Ls', Ls.cpu().data.numpy())
            train_logger.update('Lm', Lm.cpu().data.numpy())
            train_logger.update('Lm_mse', Lm_mse.cpu().data.numpy())

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))


            utils.log_to_txt(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)),
                opt['logs']['ckpt_save_path']+ opt['model']['name'] + "_" + opt['dataset']['datatype'] +".txt"
            )

def validate(val_loader, model):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()

    if not os.path.exists("cache"): os.mkdir("cache")

    if os.path.exists("cache/{}_val.npy".format(len(val_loader.dataset))):
        info = pickle.load(open("cache/{}_val.npy".format(len(val_loader.dataset)), 'rb'))
        input_visual = info['input_visual']
        input_text = info['input_text']
        input_audio = info['input_audio']
        queried_visual = info['queried_visual']
        input_text_lengeth = info['input_text_lengeth']
        del info
    else:
        input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
        queried_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
        input_audio = np.zeros((len(val_loader.dataset), 3, 256, 256))
        input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
        input_text_lengeth = [0]*len(val_loader.dataset)

        for i, val_data in enumerate(val_loader):

            images, images_2, audios, captions, lengths, ids= val_data

            for (id, img, img2, cap, aud, l) in zip(ids, (images.numpy().copy()), (images_2.numpy().copy()), (captions.numpy().copy()), (audios.numpy().copy()), lengths):
                queried_visual[id] = img2

                input_visual[id] = img
                input_text[id, :captions.size(1)] = cap
                input_text_lengeth[id] = l
                input_audio[id] = aud


        input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])
        queried_visual = np.array([queried_visual[i] for i in range(0, len(queried_visual), 5)])

        filename = "cache/{}_val.npy".format(len(val_loader.dataset))
        info = {
            "input_visual": input_visual,
            "input_text": input_text,
            "input_audio": input_audio,
            "queried_visual": queried_visual,
            "input_text_lengeth": input_text_lengeth
        }
        pickle.dump(info, open(filename, 'wb'), protocol=4)
        del info


    end = time.time()
    print("calculate similarity time:", end - start)

    ### I2T
    d = utils.shard_dis(input_visual, input_text, model , text_lens=input_text_lengeth,  mode='i2t' )

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore_vt = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score_1 = "I2T: r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore_vt
    )

    ### I2A
    d = utils.shard_dis(input_visual, input_audio, model ,  mode='i2a' )

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to audio: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Audio to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore_va = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score_2 = "I2A: r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore_va
    )


    ### I2I
    d = utils.shard_dis(input_visual, queried_visual, model ,  mode='i2i' )

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2i(d)
    logging.info("Image to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_i2i(np.transpose(d))
    logging.info("Image to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore_vv = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score_3 = "I2I: r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore_vv
    )

    currscore = (currscore_vt + currscore_va + currscore_vv) / 3.0

    return currscore, all_score_1 + all_score_2 + all_score_3

def validate_test(val_loader, model):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    print("start test ...")

    if os.path.exists("cache/{}_test.npy".format(len(val_loader.dataset))):
        info = pickle.load(open("cache/{}_test.npy".format(len(val_loader.dataset)), 'rb'))
        input_visual = info['input_visual']
        input_text = info['input_text']
        input_audio = info['input_audio']
        queried_visual = info['queried_visual']
        input_text_lengeth = info['input_text_lengeth']
        del info
    else:
        input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
        queried_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
        input_audio = np.zeros((len(val_loader.dataset), 3, 256, 256))
        input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
        input_text_lengeth = [0]*len(val_loader.dataset)

        for i, val_data in tqdm(enumerate(val_loader)):

            images, images_2, audios, captions, lengths, ids= val_data

            for (id, img, img2, cap, aud, l) in zip(ids, (images.numpy().copy()), (images_2.numpy().copy()), (captions.numpy().copy()), (audios.numpy().copy()), lengths):
                queried_visual[id] = img2

                input_visual[id] = img
                input_text[id, :captions.size(1)] = cap
                input_text_lengeth[id] = l
                input_audio[id] = aud


        mask = range(0, len(input_visual), 5)
        input_visual = input_visual[mask, :]
        queried_visual = queried_visual[mask, :]

        filename = "cache/{}_test.npy".format(len(val_loader.dataset))

        info = {
            "input_visual": input_visual,
            "input_text": input_text,
            "input_audio": input_audio,
            "queried_visual": queried_visual,
            "input_text_lengeth": input_text_lengeth
        }

        pickle.dump(info, open(filename, 'wb'), protocol=4)
        del info

    ## I2T
    d_i2t = utils.shard_dis(input_visual, input_text, model ,  mode='i2t' )
    #
    ## I2A
    d_i2a = utils.shard_dis(input_visual, input_audio, model ,  mode='i2a' )
    #
    ## I2I
    d_i2i = utils.shard_dis(input_visual, queried_visual, model ,  mode='i2i' )
    #
    return [d_i2t, d_i2a, d_i2i]
