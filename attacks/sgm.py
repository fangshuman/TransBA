import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attacker import IFGSM_Based_Attacker


def get_default_gamma(arch):
    if arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        return 0.2
    elif arch in ["densenet121", "densenet169", "densenet201"]:
        return 0.5
    else:
        return 1.

def backward_hook(gamma):
    # implement SGM through grad through ReLU
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook

def backward_hook_norm(module, grad_in, grad_out):
    # normalize the gradient to avoid gradient explosion or vanish
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


class SGM_Attacker(IFGSM_Based_Attacker):
    def get_config(arch):    
        config = super().get_config(arch)
        config['sgm_gamma'] = get_default_gamma(arch)
        return config

    def __init__(self, attach_method, model, loss_fn, args):
        super().__init__(
            attack_method=attach_method,
            model=model,
            loss_fn=loss_fn,
            args=args,
        )

        for k, v in self.default_value.items():
            self.load_params(k, v, args)

        self.register_hook()


    def load_params(self, key, value, args):
        try:
            self.__dict__[key] = vars(args)[key]
        except:
            self.__dict__[key] = value


    def register_hook(self):
        if self.arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            if self.arch in ["resnet50", "resnet101", "resnet152"]:
                # There are 2 ReLU in Conv module ResNet-50/101/152
                gamma = np.power(self.sgm_gamma, 0.5)
            else:
                gamma = self.sgm_gamma
            backward_hook_sgm = backward_hook(gamma)

            for name, module in self.model.named_modules():
                if 'relu' in name and not '0.relu' in name:
                    module.register_backward_hook(backward_hook_sgm)
                if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
                    module.register_backward_hook(backward_hook_norm)

        elif self.arch in ["densenet121", "densenet169", "densenet201"]:
            # There are 2 ReLU in Conv module DenseNet-121/169/201
            gamma = np.power(self.sgm_gamma, 0.5)
            backward_hook_sgm = backward_hook(gamma)

            for name, module in self.model.named_modules():
                if 'relu' in name and not 'transition' in name:
                    module.register_backward_hook(backward_hook_sgm)

