import torch
import torch.nn as nn
import pretrainedmodels

from .config import source_model, target_model


class Wrap(nn.Module):
    def __init__(self, arch, model):
        super(Wrap, self).__init__()
        self.model_name = arch
        self.model = model
        self.mean = self.model.mean
        self.std = self.model.std
        self.input_size = self.model.input_size

        self._mean = torch.tensor(self.mean).view(3, 1, 1).cuda()
        self._std = torch.tensor(self.std).view(3, 1, 1).cuda()

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        return self.model((x - self._mean) / self._std)


def make_model(arch):      
    if arch in ["vit", "deit"]:
        if arch == "vit":
            # https://github.com/lukemelas/PyTorch-Pretrained-ViT
            from pytorch_pretrained_vit import ViT

            model = ViT("B_16_imagenet1k", pretrained=True)
            model.mean = [0.5, 0.5, 0.5]
            model.std = [0.5, 0.5, 0.5]
            model.input_size = [384, 384]
        elif arch == "deit":
            # https://github.com/facebookresearch/deit
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
    return Wrap(arch, model)