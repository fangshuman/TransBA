import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F

from attacks.utils import normalize_by_pnorm

from .base_attacker import Based_Attacker


    
class EMI_Attacker(Based_Attacker):
    def get_config(arch):
        config = {
            "eps": 16,
            "nb_iter": 10,
            "eps_iter": 1.6,
            "prob": 0.5,  
            "kernlen": 7, 
            "nsig": 3,
            "decay_factor": 1.0,  

            "emi_sampling_interval": 7, 
            "emi_sampling_number": 11, 
        }
        return config

    def __init__(
        self,
        attack_method,
        model,
        loss_fn,
        args,
    ):
        default_value = {
            # basic default value
            "eps": 0.05,
            "nb_iter": 10,
            "eps_iter": 0.005,
            "target": False,
            # extra default value
            "prob": 0.5,
            "kernlen": 7,
            "nsig": 3,
            "decay_factor": 1.0,
            # for EMI
            "emi_sampling_interval": 7, 
            "emi_sampling_number": 11, 
        }
        for k, v in default_value.items():
            self.load_params(k, v, args)
        
        super().__init__(
            attack_method=attack_method,
            model=model,
            loss_fn=loss_fn,
            args=args
        )
        

    def load_params(self, key, value, args):
        try:
            self.__dict__[key] = vars(args)[key]
        except:
            self.__dict__[key] = value

    def perturb(self, x, y):
        eps_iter = self.eps_iter
        delta = torch.zeros_like(x)
        delta.requires_grad_()

        factors = np.linspace(
            -self.emi_sampling_interval, self.emi_sampling_interval, self.emi_sampling_number
        )

        lgrad = torch.zeros_like(x)
        g = torch.zeros_like(x)

        for i in range(self.nb_iter):
            img_x = x + delta

            img_x = torch.cat(
                [img_x + f * eps_iter * lgrad for f in factors],
                dim=0,
            )
            if i == 0:
                y = y.repeat(
                    int(self.emi_sampling_number),
                )

            outputs = self.model(img_x)
            loss = self.loss_fn(outputs, y)
            loss.backward()
            grad = delta.grad.data

            # import pdb; pdb.set_trace()
            lgrad = grad / torch.abs(grad).mean([1, 2, 3], keepdim=True)

            g = self.decay_factor * g + lgrad

            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x

            delta.grad.data.zero_()

        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        return x_adv

