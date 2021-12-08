import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import normalize_by_pnorm
from .base import Attack


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


class SGM_Attacker(Attack):
    def get_config(arch):
        config = {
            'eps': 16,
            'nb_iter': 10,
            'eps_iter': 1.6,
            'sgm_gamma': get_default_gamma(arch),
        }
        return config

    def __init__(self, attack_method, model, loss_fn, args):
        self.arch = args.source_model
        self.model = model
        self.loss_fn = loss_fn
        self.attack_method = args.attack_method

        default_value = {
            # basic default value
            'eps': 0.05,
            'nb_iter': 10,
            'eps_iter': 0.005,
            'target': False,
            'sgm_gamma': 0.5,
            # extra default value
            'decay_factor': 1.0,
            'prob': 0.5,
            'kernlen': 7,
            'nsig': 3,
        }
        for k, v in default_value.items():
            self.load_params(k, v, args)

        self.register_hook()


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
            if "di" in self.attack_method:
                outputs = self.model(self.input_diversity(x + delta))
            else:
                outputs = self.model(x + delta)

            loss = self.loss_fn(outputs, y)
            if self.target:
                loss = -loss

            loss.backward()
            grad = delta.grad.data

            # Gaussian kernel: TI-FGSM
            if "ti" in self.attack_method:
                kernel = self.get_Gaussian_kernel(x)
                grad = F.conv2d(grad, kernel, padding=self.kernlen // 2)

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


    def input_diversity(self, img):
        size = img.size(2)
        resize = int(size / 0.875)

        gg = torch.rand(1).item()
        if gg >= self.prob:
            return img
        else:
            rnd = torch.randint(size, resize + 1, (1,)).item()
            rescaled = F.interpolate(img, (rnd, rnd), mode="nearest")
            h_rem = resize - rnd
            w_hem = resize - rnd
            pad_top = torch.randint(0, h_rem + 1, (1,)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_hem + 1, (1,)).item()
            pad_right = w_hem - pad_left
            padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
            padded = F.interpolate(padded, (size, size), mode="nearest")
            return padded

    def get_Gaussian_kernel(self, x):
        # define Gaussian kernel
        kern1d = st.norm.pdf(np.linspace(-self.nsig, self.nsig, self.kernlen))
        kernel = np.outer(kern1d, kern1d)
        kernel = kernel / kernel.sum()
        kernel = torch.FloatTensor(kernel).expand(x.size(1), x.size(1), self.kernlen, self.kernlen)
        kernel = kernel.to(x.device)
        return kernel
