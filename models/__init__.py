import torch
import torch.nn as nn

from .torchvision_models import *
from .inceptionv4 import inceptionv4
from .inceptionresnetv2 import inceptionresnetv2


def wrap(model):
    mean = torch.tensor(model.mean).view(3, 1, 1)
    std = torch.tensor(model.std).view(3, 1, 1)
    model._forward = model.forward
    model.forward = lambda x: model._forward((x - mean.to(x.device)) / std.to(x.device))
    return model


def make_model(arch):
    if arch == 'vgg16':
        model = vgg16_bn(num_classes=1000, pretrained='imagenet')
    elif arch == 'resnet50':
        model = resnet50(num_classes=1000, pretrained='imagenet')
    elif arch == 'resnet101':
        model = resnet101(num_classes=1000, pretrained='imagenet')
    elif arch == 'resnet152':
        model = resnet152(num_classes=1000, pretrained='imagenet')
    elif arch == 'densenet121':
        model = densenet121(num_classes=1000, pretrained='imagenet')
    elif arch == 'densenet169':
        model = densenet169(num_classes=1000, pretrained='imagenet')
    elif arch == 'densenet201':
        model = densenet201(num_classes=1000, pretrained='imagenet')
    elif arch == 'inceptionv3':
        model = inceptionv3(num_classes=1000, pretrained='imagenet')
    elif arch == 'inceptionv4':
        model = inceptionv4(num_classes=1000, pretrained='imagenet')
    elif arch == 'inceptionresnetv2':
        model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
    else:
        raise NotImplementedError(f"No such model: {arch}")                       

    return wrap(model)
    #return model


if __name__ == "__main__":
    model = make_model("vgg16")
    print(model.input_size)
    print(model.mean)
    import ipdb
    ipdb.set_trace()

