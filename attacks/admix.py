import numpy as np
import torch
import torch.nn.functional as F

from .base_attacker import Based_Attacker

    
class Admix_Attacker(Based_Attacker):
    def get_config(arch):
        config = {
            "eps": 16,
            "nb_iter": 10,
            "eps_iter": 1.6,
            "prob": 0.5,  
            "kernlen": 7, 
            "nsig": 3,
            "decay_factor": 1.0,  

            "admix_m1": 5,          # for Admix
            "admix_m2": 3,          # for Admix
            "admix_portion": 0.2,   # for Admix
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
            # for Admix
            "admix_m1": 5,   
            "admix_m2": 3,   
            "admix_portion": 0.2,
        }
        for k, v in default_value.items():
            self.load_params(k, v, args)
        
        super().__init__(
            attack_method=attack_method,
            model=model,
            loss_fn=loss_fn,
            args=args
        )
        
        self.mediate_grad   = None
        self.mediate_output = None
        

    def load_params(self, key, value, args):
        try:
            self.__dict__[key] = vars(args)[key]
        except:
            self.__dict__[key] = value

    def perturb(self, x, y):
        eps_iter = self.eps_iter

        # initialize extra var
        if "ti" in self.attack_method.split("_"):
            kernel = self.get_Gaussian_kernel(kernlen=self.kernlen, nsig=self.nsig)
        if "mi" in self.attack_method.split("_") or "ni" in self.attack_method.split("_"): 
            g = torch.zeros_like(x)

        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            # Admix
            grad = torch.zeros_like(x)
            for _ in range(self.admix_m2):
                indices = np.arange(0, x.shape[0])
                np.random.shuffle(indices)

                x_admix = (x + delta) + self.admix_portion * (x[indices] + delta[indices])

                # import ipdb; ipdb.set_trace()
                x_batch = torch.cat([
                    x_admix * (1.0 / pow(2, si))
                    for si in range(self.admix_m1)
                ], dim=0)

                if "di" in self.attack_method.split("_"):
                    x_batch = self.input_diversity(x_batch)

                outputs = self.model(x_batch)
        
                loss = self.loss_fn(outputs, y.repeat(self.admix_m1,))
                loss.backward()

                # import ipdb; ipdb.set_trace()
                grad += delta.grad.data
                delta.grad.data.zero_()
        
            # Gaussian kernel: TI-FGSM
            if "ti" in self.attack_method.split("_"):
                grad = self.kernel_conv(grad, kernel, kern_size=self.kernlen//2, groups=3)

            # momentum: MI-FGSM / NI-FGSM
            if "mi" in self.attack_method.split("_") or "ni" in self.attack_method.split("_"):
                # import ipdb; ipdb.set_trace()
                g = self.decay_factor * g + grad / torch.abs(grad).sum([1,2,3], keepdim=True)
                grad = g

            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x

            delta.grad.data.zero_()

        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        return x_adv
