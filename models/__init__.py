import torch
import torch.nn as nn

import pretrainedmodels  # for ImageNet
from . import cifar10_models  # for CIFAR-10
from .deit import *

# source model
ImageNet_source_model = {
    "vgg16_bn": {"source_model": "vgg16_bn", "batch_size": 32},
    "vgg16": {"source_model": "vgg16", "batch_size": 32},
    "vit": {"source_model": "vit", "batch_size": 32},
    "deit": {"source_model": "deit", "batch_size": 32},
    "resnet50": {"source_model": "resnet50", "batch_size": 80},
    "resnet152": {"source_model": "resnet152", "batch_size": 50},
    "densenet121": {"source_model": "densenet121", "batch_size": 32},
    "densenet201": {"source_model": "densenet201", "batch_size": 32},
    "inceptionv3": {"source_model": "inceptionv3", "batch_size": 64},
    "inceptionv4": {"source_model": "inceptionv4", "batch_size": 32},
    "inceptionresnetv2": {"source_model": "inceptionresnetv2", "batch_size": 32},
}
ImageNet_target_model = {
    "vit": 100,
    "deit": 100,
    "vgg16_bn": 100,
    "vgg19_bn": 100,
    "vgg16": 100,
    "vgg19": 100,
    "resnet50": 250,
    "resnet101": 250,
    "resnet152": 250,
    "densenet121": 250,
    "densenet169": 250,
    "densenet201": 250,
    "inceptionv3": 250,
    "inceptionv4": 250,
    "inceptionresnetv2": 250,
    "robust_models": 100,
}

CIFAR10_source_model = {
    "resnet50": {"source_model": "resnet50", "batch_size": 1000},
    "densenet121": {"source_model": "densenet121", "batch_size": 500},
}

CIFAR10_target_model = {
    "googlenet": 1000,
    "vgg11_bn": 1000,
    "vgg13_bn": 1000,
    "vgg16_bn": 1000,
    "vgg19_bn": 1000,
    "resnet18": 2000,
    "resnet34": 2000,
    "resnet50": 2000,
    "densenet121": 2000,
    "densenet169": 2000,
    "inceptionv3": 2000,
    "mobilenetv2": 2000,
}


class Wrap(nn.Module):
    def __init__(self, model):
        super(Wrap, self).__init__()
        self.model = model
        self.mean = self.model.mean
        self.std = self.model.std
        self.input_size = self.model.input_size

        self._mean = torch.tensor(self.mean).view(3, 1, 1).cuda()
        self._std = torch.tensor(self.std).view(3, 1, 1).cuda()

    def forward(self, x):
        return self.model.forward((x - self._mean) / self._std)


def make_model(arch, dataset="ImageNet"):
    assert dataset in ["ImageNet", "CIFAR10"]
    if dataset == "ImageNet":
        if arch in ["vit", "deit"]:
            if arch == "vit":
                from pytorch_pretrained_vit import ViT

                model = ViT("B_16_imagenet1k", pretrained=True)
                model.mean = [0.5, 0.5, 0.5]
                model.std = [0.5, 0.5, 0.5]
                model.input_size = [384, 384]
            elif arch == "deit":
                model = torch.hub.load(
                    "facebookresearch/deit:main",
                    "deit_base_patch16_224",
                    pretrained=True,
                )
                model.mean, model.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                model.input_size = [224, 224]
        else:
            model = pretrainedmodels.__dict__[arch](
                num_classes=1000, pretrained="imagenet"
            )
        return Wrap(model)
    elif dataset == "CIFAR10":
        return cifar10_models.make_model(arch)


def get_model_config(dataset, is_source=True):
    if dataset == "CIFAR10":
        return CIFAR10_source_model if is_source else CIFAR10_target_model
    else:
        return ImageNet_source_model if is_source else ImageNet_target_model
