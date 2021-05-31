import ipdb
import numpy as np
import torch
import torch.nn as nn

from .utils import normalize_by_pnorm


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


class SGM_Attack(object):
    def __init__(self, arch, model, loss_fn, args):
        self.arch = arch
        self.model = model
        self.loss_fn = loss_fn
        self.attack_method = args.attack_method

        default_value = {
            # basic default value
            'eps': 0.05,
            'nb_iter': 10, 
            'eps_iter': 0.005,
            'target': False,
            'gamma': 0.5,
            # extra default value
            'decay_factor': 1.0
        }
        for k, v in default_value.items():
            self.load_params(k, v, args)

    def load_params(self, key, value, args):
        try:
            self.__dict__[key] = vars(args)[key]
        except:
            self.__dict__[key] = value
     
    
    def perturb(self, x, y):
        if "mi" in self.attack_method:
            g = torch.zeros_like(x)

        delta = torch.zeros_like(x)
        delta.requires_grad_()
    
        for i in range(self.nb_iter):                
            outputs = self.model(x + delta)
            loss = self.loss_fn(outputs, y)
            if self.target:
                loss = -loss
        
            loss.backward()
            grad = delta.grad.data

            # momentum: MI-FGSM
            if "mi" in self.attack_method:
                g = self.decay_factor * g + normalize_by_pnorm(grad, p=1)
                grad = g

            grad_sign = grad.data.sign()
            delta.data = delta.data + self.eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0., 1.) - x

            delta.grad.data.zero_()

        x_adv = torch.clamp(x + delta, 0., 1.)
        return x_adv


    def register_hook(self):
        if self.arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            if self.arch in ["resnet50", "resnet101", "resnet152"]:
                # There are 2 ReLU in Conv module ResNet-50/101/152
                gamma = np.power(self.gamma, 0.5)
            else:
                gamma = self.gamma
            backward_hook_sgm = backward_hook(gamma)

            for name, module in self.model.named_modules():
                if 'relu' in name and not '0.relu' in name:
                    module.register_backward_hook(backward_hook_sgm)
                if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
                    module.register_backward_hook(backward_hook_norm)

        elif self.arch in ["densenet121", "densenet169", "densenet201"]:
            # There are 2 ReLU in Conv module DenseNet-121/169/201
            gamma = np.power(self.gamma, 0.5)
            backward_hook_sgm = backward_hook(gamma)

            for name, module in self.model.named_modules():
                if 'relu' in name and not 'transition' in name:
                    module.register_backward_hook(backward_hook_sgm)




