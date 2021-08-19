import torch
import torch.nn as nn

from . import googlenet 
from . import vgg
from . import resnet
from . import densenet
from . import inception
from . import mobilenetv2


class Wrap(nn.Module):
    def __init__(self, model):
        super(Wrap, self).__init__()
        self.model = model
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std  = [0.2471, 0.2435, 0.2616]
        self.input_size = [3, 32, 32]

        self._mean = torch.tensor(self.mean).view(3,1,1).cuda()
        self._std = torch.tensor(self.std).view(3,1,1).cuda()

    def forward(self, x):
        return self.model.forward((x - self._mean) / self._std)


def make_model(arch):      
    if "googlenet" in arch:
        model = getattr(googlenet, arch)(pretrained=True)
    elif "vgg" in arch:
        model = getattr(vgg, arch)(pretrained=True)
    elif "resnet" in arch:
        model = getattr(resnet, arch)(pretrained=True)
    elif "densenet" in arch:
        model = getattr(densenet, arch)(pretrained=True)
    elif "inception" in arch:
        model = getattr(inception, arch)(pretrained=True)
    elif "mobilenet" in arch:
        model = getattr(mobilenetv2, arch)(pretrained=True)
    else:
        raise NotImplementedError(f"No such cifar model: {arch}")
    return Wrap(model)

