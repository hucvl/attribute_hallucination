from skimage import io, transform
import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import scipy.io as sio
import PIL
import torchvision.transforms.functional as TF
import random

class SGNDataset(data.Dataset):
    def __init__(self, args):
        super(SGNDataset, self).__init__()
        self.img_root = args.img_root
        self.isEnhancer = args.isEnhancer
        self.image_list = open(self.img_root + "/train_images.txt").readlines()
        self.attribute_list = open(self.img_root + "/train_attributes.txt").readlines()
        self.segmentation_list = open(self.img_root + "/train_segmentations.txt").readlines()
        self.nnsegmentation_list = open(self.img_root + "/train_nnsegmentations.txt").readlines()

        self.data = self._load_dataset()

    def _load_dataset(self):
        output = []
        images = self.image_list
        i = 0
        for i, img_path in enumerate(images):

            output.append({
                'img': self.img_root + self.image_list[i][:-1],
                'att': self.img_root + self.attribute_list[i][:-1],
                'seg': self.img_root + self.segmentation_list[i][:-1],
                'nnseg': self.img_root + self.nnsegmentation_list[i][:-1]
            })
            i = i + 1
            print(str(i) + " of" + str(len(images)) + "\n")
        return output

    def _colorencode(self, category_im):
        colorcodes = sio.loadmat(self.img_root + "/color150.mat")
        colorcodes = colorcodes['colors']
        idx = np.unique(category_im)
        h, w = category_im.shape
        colorCodeIm = np.zeros((h, w, 3)).astype(np.uint8)
        for i in range(idx.shape[0]):
            if idx[i] == 0:
                continue
            b = np.where(category_im == idx[i])
            rgb = colorcodes[idx[i] - 1]
            bgr = rgb[::-1]
            colorCodeIm[b] = bgr
        return colorCodeIm

    def _binaryencode(self, category_im):
        binarycodes = sio.loadmat(self.img_root + "/binarycodes.mat")
        binarycodes = binarycodes['binarycodes']
        idx = np.unique(category_im)
        h, w = category_im.shape
        binaryCodeIm = np.zeros((h, w, 8)).astype(np.uint8)
        for i in range(idx.shape[0]):
            if idx[i] == 0:
                continue
            b = np.where(category_im == idx[i])
            binaryCodeIm[b] = binarycodes[idx[i] - 1]
        return binaryCodeIm

    def transform(self, image, seg, nnseg):
        # Resize
        resize = transforms.Resize(512)
        image = resize(image)
        resize = transforms.Resize(512, interpolation=PIL.Image.NEAREST)
        seg = resize(seg)
        resize = transforms.Resize(512, interpolation=PIL.Image.NEAREST)
        nnseg = resize(nnseg)

        # Random crop
        #i, j, h, w = transforms.RandomCrop.get_params(
        #    image, output_size=(256, 256))
        #image = TF.crop(image, i, j, h, w)
        #seg = TF.crop(seg, i, j, h, w)

        # Center crop
        crop = transforms.CenterCrop((512, 512))
        image = crop(image)
        seg = crop(seg)
        nnseg = crop(nnseg)
        if not self.isEnhancer:
            # Resize
            resize2 = transforms.Resize(256)
            image = resize2(image)
            resize2 = transforms.Resize(256, interpolation=PIL.Image.NEAREST)
            seg = resize2(seg)
            resize2 = transforms.Resize(256, interpolation=PIL.Image.NEAREST)
            nnseg = resize2(nnseg)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            seg = TF.hflip(seg)
            nnseg = TF.hflip(nnseg)


        # Random vertical flipping
        #if random.random() > 0.5:
        #    image = TF.vflip(image)
        #    seg = TF.vflip(seg)

        cat = np.array(seg)
        #seg = self._colorencode(cat)
        seg = self._binaryencode(cat)
        seg = np.transpose(seg, (2, 0, 1))

        catnn = np.array(nnseg)
        #seg = self._colorencode(cat)
        nnseg = self._binaryencode(catnn)
        nnseg = np.transpose(nnseg, (2, 0, 1))

        # Transform to tensor
        image = TF.to_tensor(image)
        seg = torch.tensor(seg)
        nnseg = torch.tensor(nnseg)
        cat = torch.tensor(cat)
        return image, seg, cat, nnseg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        img = Image.open(datum['img'])

        att = np.load(datum['att']).astype(np.float32)
        seg = Image.open(datum['seg'])
        nnseg = Image.open(datum['nnseg'])
        img, seg, cat, nnseg = self.transform(img, seg, nnseg)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)

        return img, att, seg, cat, nnseg


class SGNDatasetTest(data.Dataset):
    def __init__(self, args):
        super(SGNDatasetTest, self).__init__()
        self.img_root = args.img_root
        self.isEnhancer = args.isEnhancer
        self.image_list = open(self.img_root + "/val_images.txt").readlines()
        self.attribute_list = open(self.img_root + "/val_attributes.txt").readlines()
        self.segmentation_list = open(self.img_root + "/val_segmentations.txt").readlines()

        self.data = self._load_dataset()

    def _load_dataset(self):
        output = []
        images = self.image_list
        i = 0
        for i, img_path in enumerate(images):

            output.append({
                'img': self.img_root + self.image_list[i][:-1],
                'att': self.img_root + self.attribute_list[i][:-1],
                'seg': self.img_root + self.segmentation_list[i][:-1]
            })
            i = i + 1
            print(str(i) + " of" + str(len(images)) + "\n")
        return output

    def _colorencode(self, category_im):
        colorcodes = sio.loadmat(self.img_root + "./color150.mat")
        colorcodes = colorcodes['colors']
        idx = np.unique(category_im)
        h, w = category_im.shape
        colorCodeIm = np.zeros((h, w, 3)).astype(np.uint8)
        for i in range(idx.shape[0]):
            if idx[i] == 0:
                continue
            b = np.where(category_im == idx[i])
            rgb = colorcodes[idx[i] - 1]
            bgr = rgb[::-1]
            colorCodeIm[b] = bgr
        return colorCodeIm

    def _binaryencode(self, category_im):
        binarycodes = sio.loadmat(self.img_root + "binarycodes.mat")
        binarycodes = binarycodes['binarycodes']
        idx = np.unique(category_im)
        h, w = category_im.shape
        binaryCodeIm = np.zeros((h, w, 8)).astype(np.uint8)
        for i in range(idx.shape[0]):
            if idx[i] == 0:
                continue
            b = np.where(category_im == idx[i])
            binaryCodeIm[b] = binarycodes[idx[i] - 1]
        return binaryCodeIm

    def transform(self, image, seg):
        # Resize
        resize = transforms.Resize(512)
        image = resize(image)
        resize = transforms.Resize(512, interpolation=PIL.Image.NEAREST)
        seg = resize(seg)

        # Center crop
        crop = transforms.CenterCrop((512, 512))
        image = crop(image)
        seg = crop(seg)
      

        if not self.isEnhancer:
            # Resize
            resize2 = transforms.Resize(256)
            image = resize2(image)
            resize2 = transforms.Resize(256, interpolation=PIL.Image.NEAREST)
            seg = resize2(seg)

        cat = np.array(seg)
        #seg = self._colorencode(cat)
        seg = self._binaryencode(cat)
        seg = np.transpose(seg, (2, 0, 1))

        # Transform to tensor
        image = TF.to_tensor(image)
        seg = torch.tensor(seg)
        cat = torch.tensor(cat)
        return image, seg, cat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        img = Image.open(datum['img'])

        att = np.load(datum['att']).astype(np.float32)
        seg = Image.open(datum['seg'])
        img, seg, cat = self.transform(img, seg)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)

        return img, att, seg, cat
