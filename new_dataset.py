import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import os
import json
import cv2


class importDataset(Dataset):

    def __init__(self, root=None, label_dict = None, transform=None, target_transform=None):
        self.key = os.listdir(root)
        self.root = root
        nSamples = len(os.listdir(root))
        self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform
        with open(label_dict, 'r') as j:
            self.label_dict = json.loads(j.read())
        

    def __len__(self):
        return self.nSamples
    
    def get_label(self):
          return self.label_dict

    def get_dir(self):
        return self.dir

    def get_key(self):
        return self.key



    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # index += 1


        img = cv2.imread(os.path.join(self.root, self.key[index]), cv2.IMREAD_GRAYSCALE)
        label = self.label_dict[self.key[index]]
        
        if self.transform is not None:
            img = self.transform(img)


        if self.target_transform is not None:
            label = self.target_transform(label)
        

        return (img, label)
        

class resizeNormalize(object):

    def __init__(self, size):
        self.size = size
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = cv2.resize(img, self.size)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            # batch_index = random_start + torch.arange(0, self.batch_size - 1)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            # tail_index = random_start + torch.arange(0, tail - 1)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:

                w, h = image.shape[1], image.shape[0]
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

          
        return images, labels
