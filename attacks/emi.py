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
        delta = torch.zeros_like(x)
        # delta.requires_grad_()

        # factors = np.linspace(
        #     -self.emi_sampling_interval, self.emi_sampling_interval, self.emi_sampling_number
        # )
        
        if "ti" in self.attack_method:
            self.kernel = self.get_Gaussian_kernel(kernlen=self.kernlen, nsig=self.nsig)

        grad = torch.zeros_like(x)
        g = torch.zeros_like(x)

        for i in range(self.nb_iter):
            img_x = x + delta

            # if "di" in self.attack_method:
            #     img_x = self.input_diversity(img_x)

            # img_x = torch.cat(
            #     [img_x + f * eps_iter * lgrad for f in factors],
            #     dim=0,
            # )
            # if i == 0:
            #     y = y.repeat(
            #         int(self.emi_sampling_number),
            #     )

            cur_grad = torch.zeros_like(x)
            for _ in range(self.emi_sampling_number):
                r = torch.zeros_like(x)
                r.requires_grad_()

                coef = (torch.rand(1).item() - 0.5) * 2 * self.emi_sampling_interval/255.

                if "di" in self.attack_method:
                    outputs = self.model(self.input_diversity((img_x + coef*grad).data + r))
                else:
                    outputs = self.model((img_x + coef*grad).data + r)

                loss = self.loss_fn(outputs, y)
                loss.backward()

                cur_grad += r.grad.data.clone()
                r.grad.data.zero_()
            grad = cur_grad * (1.0/self.emi_sampling_number)

            # Gaussian kernel: TI-FGSM
            if "ti" in self.attack_method:
                grad = self.kernel_conv(grad, self.kernel, kern_size=self.kernlen//2, groups=3)

            # import pdb; pdb.set_trace()
            g = self.decay_factor * g + grad / torch.abs(grad).mean([1, 2, 3], keepdim=True)
            grad = g

            grad_sign = grad.data.sign()
            delta.data = delta.data + self.eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x

            # delta.grad.data.zero_()

        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        return x_adv

