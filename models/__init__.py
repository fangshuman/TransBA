import torch
import torch.nn as nn
import pretrainedmodels

class Wrap(nn.Module):
    def __init__(self, model):
        super(Wrap, self).__init__()
        self.model = model
        self.mean = self.model.mean
        self.std = self.model.std
        self.input_size = self.model.input_size

        self._mean = torch.tensor(self.mean).view(3,1,1).cuda()
        self._std = torch.tensor(self.std).view(3,1,1).cuda()

    def forward(self, x):
        return self.model.forward((x - self._mean) / self._std)


def make_model(arch):      
    model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
    return Wrap(model)

