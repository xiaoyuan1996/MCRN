#encoding:utf-8
# -----------------------------------------------------------
# "MCRN: A Multi-source Cross-modal Retrieval Network for Remote Sensing"
# Zhiqiang Yuan, Changyuan Tian, Wenkai Zhang, Yongqiang Mao, Hongqi Wang, and Xian Sun.
# Undergoing review.
# Writen by YuanZhiqiang, 2022.  Our code is depended on AMFMN
# ------------------------------------------------------------

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import yaml
import argparse
import utils
from vocab import deserialize_vocab
from PIL import Image
import random


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, vocab, opt):
        self.vocab = vocab
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']
        self.audio_path = opt['dataset']['audio_path']
        self.data_split = data_split

        # Captions
        self.maxlength = 0

        if data_split != 'test':
            self.audios = []
            with open(self.loc + '%s_audios_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.audios.append(line.strip())

            self.captions = []
            with open(self.loc + '%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            self.audios = []
            with open(self.loc + '%s_audios.txt' % data_split, 'rb') as f:
                for line in f:
                    self.audios.append(line.strip())

            self.captions = []
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            # images
            self.transform_i1 = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation((0, 90)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            self.transform_i2 = transforms.Compose([
                transforms.RandomRotation((0, 90)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize((278, 278)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

            # audios
            self.transform_a = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform_ia = transforms.Compose([
                transforms.RandomRotation((0, 90)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize((278, 278)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

            self.transform_i = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])


            # audios
            self.transform_a = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])


    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        caption = self.captions[index]
        audio = self.audios[index]

        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8'))
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]

        # caption
        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)

        if self.data_split == "train":
            # image
            image = Image.open(os.path.join(self.img_path, str(self.images[img_id])[2:-1])).convert('RGB')
            image_1 = self.transform_i1(image)  # torch.Size([3, 256, 256])
            image_2 = self.transform_i2(image)  # torch.Size([3, 256, 256])
        else:
            # image
            image = Image.open(os.path.join(self.img_path, str(self.images[img_id])[2:-1])).convert('RGB')
            image_1 = self.transform_i(image)  # torch.Size([3, 256, 256])
            image_2 = self.transform_ia(image)  # torch.Size([3, 256, 256])

        # audio
        audio = Image.open(os.path.join(self.audio_path, str(audio)[2:-1])).convert('RGB')
        audio = self.transform_a(audio)  # torch.Size([3, 256, 256])

        # return image, caption, tokens_UNK, index, img_id
        return image_1, image_2, audio, caption, index

    def __len__(self):
        return self.length


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[3]), reverse=True)
    image_1, image_2, audio, caption, ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    image_1 = torch.stack(image_1, 0)
    image_2 = torch.stack(image_2, 0)

    # Merge audios (convert tuple of 3D tensor to 4D tensor)
    audio = torch.stack(audio, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in caption]
    targets = torch.zeros(len(caption), max(lengths)).long()
    for i, cap in enumerate(caption):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    lengths = [l if l != 0 else 1 for l in lengths]

    return image_1, image_2, audio, targets, lengths, ids


def get_precomp_loader(data_split, vocab, batch_size=100,
                       shuffle=True, num_workers=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(vocab, opt):
    train_loader = get_precomp_loader('train', vocab,
                                      opt['dataset']['batch_size'], True, opt['dataset']['workers'], opt=opt)
    val_loader = get_precomp_loader('val', vocab,
                                    opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return train_loader, val_loader


def get_test_loader(vocab, opt):
    test_loader = get_precomp_loader('test', vocab,
                                     opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return test_loader
