import os
import logging
from glob import glob

import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, phase, total=None, size=224):
        assert phase in ["cln", "adv"]
        self.image_dir = image_dir
        # self.class_to_idx = np.load(label_dir, allow_pickle=True)[()]
        self.image_list = os.listdir(image_dir)
        self.image_list.sort()

        if phase == "cln":
            self.image_list = [item for item in self.image_list if "png" in item]
            self.transform = transforms.Compose(
                [
                    # transforms.Resize(int(size / 0.875)),
                    # transforms.CenterCrop(size),
                    transforms.Resize(size),
                    transforms.ToTensor(),
                ]
            )

        elif phase == "adv":
            self.image_list = [item for item in self.image_list if "png" in item]
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.ToTensor()
                ]
            )

        # assert len(self.image_list) == total
        if total is None:
            total = len(self.image_list)

        if len(self.image_list) > total:
            logging.warning(f"Attack {total} images with total: {len(self.image_list)}")
            self.image_list = self.image_list[:total]

    def __getitem__(self, index):
        image_path = self.image_list[index]
        with open(os.path.join(self.image_dir, image_path), "rb") as f:
            with Image.open(f) as image:
                image = image.convert("RGB")
        image = self.transform(image)
        # label = self.class_to_idx[self.image_list[index].split(".")[0].split("_")[0]]
        label = -1
        return image, label, index

    def __len__(self):
        return len(self.image_list)


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, total=None, size=224):
        image_dirname = os.path.join(image_dir, "val")
        classes = [
            d
            for d in os.listdir(image_dirname)
            if os.path.isdir(os.path.join(image_dirname, d))
        ]
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        assert len(self.class_to_idx) == 1000

        self.images_fname = glob("{}/*/*.JPEG".format(image_dirname))
        self.image_list = [
            fname.split("/")[-2] + "_" + fname.split("/")[-1]
            for fname in self.images_fname
        ]

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(size / 0.875)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
        )

        # assert len(self.image_list) == total
        if total is None:
            total = len(self.images_fname)

        if len(self.images_fname) > total:
            logging.warning(f"Attack {total} images with total: {len(self.images_fname)}")
            self.images_fname = self.images_fname[:total]
            self.image_list = self.image_list[:total]

    def __getitem__(self, index):
        image_path = self.images_fname[index]
        with open(image_path, "rb") as f:
            with Image.open(f) as image:
                image = image.convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[self.images_fname[index].split("/")[-2]]
        return image, label, index

    def __len__(self):
        return len(self.images_fname)



def make_loader(image_dir, label_dir, phase, batch_size=1, total=None, size=224):
    """
    Args:
        image_dir: image root directory
        label_dir: label path, if None using ImageFolderDataset
        batch_size: input batch size for adversarial attack
        total: the number of images to be attacked
        size: the size of input images
    Return:
        list, dataloader
    """
    if label_dir is True:
        imgset = ImageFolderDataset(image_dir, total=total, size=size)
    else:
        imgset = ImageNetDataset(image_dir, phase, total=total, size=size)
    loader = torch.utils.data.DataLoader(imgset, batch_size=batch_size)
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
        image.save(os.path.join(output_dir, img_list[index].split(".")[0] + ".png"))
