import os
import logging

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, total=None, size=224):
        self.image_dir = image_dir
        self.class_to_idx = np.load(label_dir, allow_pickle=True)[()]
        self.image_list = os.listdir(image_dir)
        self.image_list.sort()
        self.image_list = self.image_list[:total]
        self.image_list = [item for item in self.image_list if "png" in item]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

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
        label = self.class_to_idx[self.image_list[index].split(".")[0].split("_")[0]]
        return image, label, index

    def __len__(self):
        return len(self.image_list)


def make_loader(image_dir, label_dir, batch_size=1, total=None, size=224):
    imgset = ImageNetDataset(image_dir, label_dir, total=total, size=size)
    loader = torch.utils.data.DataLoader(imgset, batch_size=batch_size)
    return imgset.image_list, loader


def save_image(images, indexs, img_list, output_dir):
    for i, index in enumerate(indexs):
        cur_image = (images[i, :, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)
        image = Image.fromarray(cur_image)
        image.save(os.path.join(output_dir, img_list[index].split(".")[0] + ".png"))
