"""
@date 2024/4/16 14:27
"""
import numbers

import mxnet as mx
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
import os
import random


class FaceDataset(Dataset):

    def __init__(self, root_dir: str, transform: object = None):
        super(FaceDataset, self).__init__()
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            ])
        self.imgrec = mx.recordio.MXIndexedRecordIO(
            os.path.join(root_dir, 'train.idx'),
            os.path.join(root_dir, 'train.rec'), 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        _, img = mx.recordio.unpack(s)
        image = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            image = self.transform(image)
        # 水平旋转
        flip_image = F.hflip(image)
        return image, flip_image

    def __len__(self):
        # return 2000
        return len(self.imgidx)


class MXFaceDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super(MXFaceDataset, self).__init__()
        if not transform:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            ])
            self.flag = False
        else:
            self.transform = transform
            self.flag = True
        self.imgrec = mx.recordio.MXIndexedRecordIO(
            os.path.join(root_dir, 'train.idx'),
            os.path.join(root_dir, 'train.rec'), 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        # 水平旋转
        if not self.flag:
            flip_flag = False
            if torch.rand(1) < 0.5:
                flip_flag = True
                sample = F.hflip(sample)
            return index, sample, flip_flag, label
        else:
            return index, sample, True, label

    def __len__(self):
        return len(self.imgidx)


class PreloadFaceDataset(Dataset):

    def __init__(
        self,
        root_dir,
    ):
        super(PreloadFaceDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.imgrec = mx.recordio.MXIndexedRecordIO(
            os.path.join(root_dir, 'train.idx'),
            os.path.join(root_dir, 'train.rec'), 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        _, img = mx.recordio.unpack(s)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, F.hflip(sample)

    def __len__(self):
        return len(self.imgidx)


def get_dataloader(root_dir,
                   batch_size,
                   num_workers=4,
                   shuffle=True,
                   transform=None):
    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')

    if os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, transform=transform)
    # Image Folder
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_set = ImageFolder(root_dir, transform)
    return DataLoader(train_set,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=True)


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomErasing(0.1,
                                 scale=(0.02, 0.33),
                                 ratio=(0.3, 3.3),
                                 value=0,
                                 inplace=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_distribute_dataloader(root_dir, rank, batch_size, num_workers=2):
    """
    使用ddp方案时,需要使用分布式的dataloader
    每个dataloader获取同一个batch的不同的部分
    """
    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')

    if os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir)
    # Image Folder
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_set = ImageFolder(root_dir, transform)
    #  arcface手动定义了DistributedSampler和torch写的没有什么区别，就是torch的可以drop_last
    sampler = DistributedSampler(train_set,
                                 shuffle=True,
                                 rank=rank,
                                 drop_last=True)
    # 每个epoch都需要重置采样器的种子，否则每个epoch采样器返回的indice采样都是一样的顺序
    # worker_init_fn产生随机数种子，dataloader会调用dataset进行数据增强，在这区间可能会需要随机数种子
    return sampler, DataLoader(train_set,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               sampler=sampler)
