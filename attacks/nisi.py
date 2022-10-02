import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F

from .base_attacker import Based_Attacker

    
class NI_SI_Attacker(Based_Attacker):
    def get_config(arch):
        config = {
            "eps": 16,
            "nb_iter": 10,
            "eps_iter": 1.6,
            "prob": 0.5,  
            "kernlen": 7, 
            "nsig": 3,
            "decay_factor": 1.0,    # for ni
            "scale_copies": 5,      # for si
        }
        return config

    def __init__(
        self,
        attack_method,
        model,
        loss_fn,
        args,
    ):
        self.attack_method = attack_method
        self.model = model
        self.loss_fn = loss_fn

        self.default_value = {
            # basic default value
            "eps": 0.05,
            "nb_iter": 10,
            "eps_iter": 0.005,
            "target": False,
            # extra default value
            "prob": 0.5,
            "kernlen": 7,
            "nsig": 3,
            # for NI
            "decay_factor": 1.0,
            # for SI
            "scale_copies": 5,
        }
        for k, v in self.default_value.items():
            self.load_params(k, v, args)

    def load_params(self, key, value, args):
        try:
            self.__dict__[key] = vars(args)[key]
        except:
            self.__dict__[key] = value

    def perturb(self, x, y):
        eps_iter = self.eps_iter

        # initialize extra var
        if "ti" in self.attack_method:
            ti_kernel = self.get_Gaussian_kernel(kernlen=self.kernlen, nsig=self.nsig)


        g = torch.zeros_like(x)

        extra_item = torch.zeros_like(x)
        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            img_x = x + delta
            # NI
            img_x = img_x + self.decay_factor * eps_iter * g
            
            # SI
            img_x = torch.cat([
                img_x * (1.0 / pow(2, si))
                for si in range(self.scale_copies)
            ], dim=0)
            if i == 0:
                y = y.repeat(self.scale_copies,)
            
            if "di" in self.attack_method:
                img_x = self.input_diversity(img_x)

            outputs = self.model(img_x)
            loss = self.loss_fn(outputs, y)
            loss.backward()

            grad = delta.grad.data

            # Gaussian kernel: TI-FGSM
            if "ti" in self.attack_method:
                grad = self.kernel_conv(grad, ti_kernel, kern_size=self.kernlen//2, groups=3)

            # momentum: NI-FGSM
            g = self.decay_factor * g + grad / torch.abs(grad).mean([1,2,3], keepdim=True)
            grad = g

            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign + extra_item
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x

            delta.grad.data.zero_()

        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        return x_adv
