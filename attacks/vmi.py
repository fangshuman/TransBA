import numpy as np
import torch
import torch.nn.functional as F

from .base_attacker import Based_Attacker

    
class VMI_Attacker(Based_Attacker):
    def get_config(arch):
        config = {
            "eps": 16,
            "nb_iter": 10,
            "eps_iter": 1.6,
            "prob": 0.5,  
            "kernlen": 7, 
            "nsig": 3,
            "decay_factor": 1.0,  

            "vmi_sample_n": 20,      # for vmi
            "vmi_sample_beta": 1.5,  # for vmi
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
            # for VMI
            "vmi_sample_n": 20,      # for vmi
            "vmi_sample_beta": 1.5,  # for vmi
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

    def init_process(self, x):
        super().init_process(x)
        self.variance = torch.zeros_like(x)

    def gradient_process(self, x, y, grad):
        global_grad = torch.zeros_like(x)
        for _ in range(self.vmi_sample_n):
            region = self.vmi_sample_beta * self.eps
            r = torch.zeros_like(x)
            r.data.uniform_(-region, region)
            r.requires_grad_()

            if "di" in self.attack_method:
                outputs = self.model(self.input_diversity(x.data + r))
            else:
                outputs = self.model(x.data + r)
            loss = self.loss_fn(outputs, y)
            loss.backward()

            global_grad += r.grad.data.clone()
            r.grad.data.zero_()

        current_grad = grad + self.variance
        grad = current_grad

        # update variance
        self.variance = global_grad * (1.0/self.vmi_sample_n) - grad

        # Gaussian kernel: TI-FGSM
        if "ti" in self.attack_method:
            grad = self.kernel_conv(grad, self.kernel, kern_size=self.kernlen//2, groups=3)

        self.g = self.decay_factor * self.g + grad / torch.abs(grad).mean([1,2,3], keepdim=True)
        grad = self.g
        return grad

    def perturb(self, x, y):
        self.init_process(x)

        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            img_x = x + delta

            outputs = self.model(img_x)
            loss = self.loss_fn(outputs, y)
            loss.backward()

            grad = delta.grad.data
            grad = self.gradient_process(img_x, y, grad=grad)

            grad_sign = grad.data.sign()
            delta.data = delta.data + self.eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x

            delta.grad.data.zero_()

        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        return x_adv
