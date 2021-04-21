import os
import csv

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from advertorch.attacks import LinfPGDAttack
class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, phase, total=1000, size=224):
        assert phase in ['cln', 'adv']
        self.image_dir = image_dir 
        self.class_to_idx = np.load(label_dir, allow_pickle=True)[()]
        self.image_list = os.listdir(image_dir)
        self.image_list.sort()
        self.phase = phase

        if phase == 'cln':
            self.image_list = [item for item in self.image_list if 'JPEG' in item]
            self.transform = transforms.Compose([
                transforms.Resize(int(size / 0.875)),
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ])
        elif phase == 'adv':
            self.image_list = [item for item in self.image_list if 'png' in item]
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()
            ])
        assert len(self.image_list) == total


    def __getitem__(self, index):
        image_path = self.image_list[index]
        with open(os.path.join(self.image_dir, image_path), 'rb') as f:
            with Image.open(f) as image:
                image = image.convert('RGB')
        image = self.transform(image)
        
        if self.phase == 'cln':
            label = self.class_to_idx[self.image_list[index].split('_')[0]]
        elif self.phase == 'adv':
            label = self.class_to_idx[index]

        return image, label, index


    def __len__(self):
        return len(self.image_list)



def make_loader(image_dir, label_dir, phase, batch_size=1, total=1000, size=224):
    """
    Args:
        image_dir: image root directory
        label_dir: label path
        batch_size: input batch size for adversarial attack
        total: the number of images to be attacked
        size: the size of input images
    Return:
        list, dataloader
    """
    imgset = ImageNetDataset(image_dir, label_dir, phase, total=total, size=size)
    loader = torch.utils.data.DataLoader(
        imgset,
        batch_size=batch_size
    )
    return imgset.image_list, loader


def save_image(images, indexs, img_list, output_dir):
    """
    Args:
        images: minibatch of images type with numpy
        img_list: list of filenames without path
        output_dir: directory where to save images
    """
    for i, index in enumerate(indexs):
        cur_image = (images[i, :, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)
        image = Image.fromarray(cur_image)
        image.save(os.path.join(output_dir, img_list[index].split('.')[0] + '.png'))
