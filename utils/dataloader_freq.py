import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.data_augmentation import cv_random_flip, randomCrop, randomRotation, randomPeper, colorEnhance
from utils.dct import dct_2d
import pickle


class TrainDataset(Dataset):
    """
    dataloader for COD tasks
    Implemented according to DGNet
    """

    def __init__(self, image_root, gt_root, trainsize, edge_root=None, rVFlip=True, rCrop=True, rRotate=True,
                 colorEnhance=True, rPeper=True):
        self.edge_root = edge_root
        self.trainsize = trainsize
        self.rVFlip = rVFlip
        self.rCrop = rCrop
        self.rRotate = rRotate
        self.colorEnhance = colorEnhance
        self.rPeper = rPeper
        self.imgs = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                     f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        if edge_root is not None:
            self.edges = [os.path.join(edge_root, f) for f in os.listdir(edge_root) if f.endswith('.jpg')
                          or f.endswith('.png')]

        # sorted files
        self.imgs = sorted(self.imgs)
        self.gts = sorted(self.gts)
        if edge_root is not None:
            self.edges = sorted(self.edges)

        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.freq_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.PILToTensor()
        ])
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        # frequency mean and sample std
        with open('./utils/freq_mean_std.pkl', 'rb') as f:
            freq_stats = pickle.load(f)
        self.freq_norm = transforms.Normalize(mean=freq_stats['mean'], std=freq_stats['std'])

        self.size = len(self.imgs)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        img = self.rgb_loader(self.imgs[index])
        gt = self.binary_loader(self.gts[index])
        if self.edge_root is not None:
            edge = self.binary_loader(self.edges[index])

        # Data Augmentation
        # random horizental flipping
        if self.edge_root is not None:
            if self.rVFlip:
                img, gt, edge = cv_random_flip([img, gt, edge])
            if self.rCrop:
                img, gt, edge = randomCrop([img, gt, edge])
            if self.rRotate:
                img, gt, edge = randomRotation([img, gt, edge])
        else:
            if self.rVFlip:
                img, gt = cv_random_flip([img, gt])
            if self.rCrop:
                img, gt = randomCrop([img, gt])
            if self.rRotate:
                img, gt = randomRotation([img, gt])
        # bright, contrast, color, sharp jitters
        if self.colorEnhance:
            img = colorEnhance(img)
        # random peper noise
        if self.rPeper:
            gt = randomPeper(gt)

        # DCT feature
        freq = self.freq_transform(img).unsqueeze(0)
        freq = dct_2d(freq).squeeze(0)
        freq = self.freq_norm(freq) / 7.0
        high, low = self.freq_decompose(freq)
        # print(freq.shape)   # torch.Size([192, 48, 48])

        # RGB feature
        img = self.img_transform(img)
        gt = self.gt_transform(gt)
        if self.edge_root is not None:
            edge = self.gt_transform(edge)
        if self.edge_root is not None:
            return img, gt, edge, high, low
        else:
            return img, gt, high, low

    def filter_files(self):
        assert len(self.imgs) == len(self.gts)
        if self.edge_root is not None:
            assert len(self.edges) == len(self.imgs)
        images = []
        gts = []
        for img_path, gt_path in zip(self.imgs, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.imgs = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def freq_decompose(self, freq):
        freq_y = freq[0:64, :, :]
        freq_Cb = freq[64:128, :, :]
        freq_Cr = freq[128:192, :, :]
        high = torch.cat([freq_y[32:, :, :], freq_Cb[32:, :, :], freq_Cr[32:, :, :]], dim=0)
        low = torch.cat([freq_y[:32, :, :], freq_Cb[:32, :, :], freq_Cr[:32, :, :]], dim=0)
        return high, low

    def __len__(self):
        return self.size


class TestDataset(Dataset):
    def __init__(self, image_root, gt_root, testsize, edge_root=None):
        self.testsize = testsize
        self.edge_root = edge_root
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        if edge_root is not None:
            self.edges = [os.path.join(edge_root, f) for f in os.listdir(edge_root) if f.endswith('.jpg')
                          or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if edge_root is not None:
            self.edges = sorted(self.edges)

        self.freq_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.PILToTensor()
        ])

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])

        # frequency mean and sample std
        with open('./utils/freq_mean_std.pkl', 'rb') as f:
            freq_stats = pickle.load(f)
        self.freq_norm = transforms.Normalize(mean=freq_stats['mean'], std=freq_stats['std'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.edge_root is not None:
            edge = self.binary_loader(self.edges[index])

        # DCT feature
        freq = self.freq_transform(image).unsqueeze(0)
        freq = dct_2d(freq).squeeze(0)
        freq = self.freq_norm(freq) / 7.0
        high, low = self.freq_decompose(freq)

        image = self.transform(image)
        gt_origin = transforms.PILToTensor()(gt)
        gt = self.gt_transform(gt)
        if self.edge_root is not None:
            edge_origin = transforms.PILToTensor()(edge)
            edge = self.gt_transform(edge)
        name = self.images[index].split('/')[-1]
        if '\\' in name:
            name = name.split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        if self.edge_root is not None:
            return image, gt, gt_origin, edge, edge_origin, name, high, low
        else:
            return image, gt, gt_origin, name, high, low

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def freq_decompose(self, freq):
        freq_y = freq[0:64, :, :]
        freq_Cb = freq[64:128, :, :]
        freq_Cr = freq[128:192, :, :]
        high = torch.cat([freq_y[32:, :, :], freq_Cb[32:, :, :], freq_Cr[32:, :, :]], dim=0)
        low = torch.cat([freq_y[:32, :, :], freq_Cb[:32, :, :], freq_Cr[:32, :, :]], dim=0)
        return high, low
