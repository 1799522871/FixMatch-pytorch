import logging
import math

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

train_dir = 'D:/codestudy/MyPythonProject/remote-sesing/FixMatch-pytorch/data/UCM_split/train/'
test_dir = 'D:/codestudy/MyPythonProject/remote-sesing/FixMatch-pytorch/data/UCM_split/test/'

normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)




from pathlib import Path
from typing import Callable, Optional, Any
import torchvision
from torchvision.datasets import VisionDataset


def get_UCM(args, root, test_dir):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=256,
                              padding=int(256 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)])

    base_dataset = datasets.ImageFolder(root=root, transform=None)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets) # base_dataset.targets 获取标签

    train_labeled_dataset = UCMSSL(
        root=root, indexs=train_labeled_idxs,
        transform=transform_labeled)

    train_unlabeled_dataset = UCMSSL(
        root=root, indexs=train_unlabeled_idxs,
        transform=TransformFixMatch_RS(mean=normal_mean, std=normal_std))

    test_dataset = UCMSSL_test(root=test_dir,transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class TransformFixMatch_RS(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=256,
                                  padding=int(256 * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=256,
                                  padding=int(256 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class UCMSSL(Dataset):
    def __init__(self, root, indexs,
                 transform=None, target_transform = None):
        basedata = datasets.ImageFolder(root,transform=None)
        list = []
        if indexs is not None:
            # for i in indexs:
            #     list.append(np.array(Image.open(basedata.imgs[i][0])))
            for i in indexs:
                image = Image.open(basedata.imgs[i][0])
                resized_image = image.resize((256, 256))
                array = np.array(resized_image)
                list.append(array)

            self.targets = np.array(basedata.targets)[indexs]
            self.classes = basedata.classes  # Add this line
            self.class_to_idx = basedata.class_to_idx
            # self.imgs = np.array(basedata.imgs)[indexs]

            # 看的Dataset.cifar10
            # self.data = np.vstack(list).reshape(-1, 3, 256, 256)
            self.data = np.stack(list, axis = 0)
            # self.data = self.data.transpose((0, 2, 3, 1))
            print('train:{}'.format(self.data.shape))

            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)

class UCMSSL_test(Dataset):
    def __init__(self, root,
                 transform=None, target_transform = None):
        basedata = datasets.ImageFolder(root,transform=None)
        list = []
        for i in range(len(basedata.imgs)):
            image = Image.open(basedata.imgs[i][0])
            resized_image = image.resize((256, 256))
            array = np.array(resized_image)
            list.append(array)


        self.targets = basedata.targets
        self.classes = basedata.classes  # Add this line
        self.class_to_idx = basedata.class_to_idx
        self.imgs = basedata.imgs

        # 看的Dataset.cifar100
        # self.data = np.vstack(list).reshape(-1, 3, 256, 256)
        self.data =  np.stack(list,axis=0)
        # self.data = self.data.transpose((0, 2, 3, 1))
        print('test:{}'.format(self.data.shape))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)


