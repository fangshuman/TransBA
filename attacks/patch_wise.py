import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F

from .base import Attack

    
class Patchwise_Attacker(Attack):
    def get_config(arch):
        config = {
            "eps": 16,
            "nb_iter": 10,
            "eps_iter": 1.6,
            "prob": 0.5,  
            "kernlen": 7, 
            "nsig": 3,
            "decay_factor": 1.0,  
            "pi_beta": 10,          # for pi (patch-wise)
            "pi_gamma": 16,         # for pi (patch-wise)
            "pi_kern_size": 3,      # for pi (patch-wise)
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
            "decay_factor": 1.0,
            "pi_beta": 10,
            "pi_gamma": 16,
            "pi_kern_size": 3,
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
        if "mi" in self.attack_method or "ni" in self.attack_method: 
            g = torch.zeros_like(x)
    
        amplification = torch.zeros_like(x)
        eps_iter *= self.pi_beta
        pi_kernel = self.get_project_kernel(kern_size=self.pi_kern_size)
        if self.pi_gamma > 1:
            self.pi_gamma /= 255.

        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            img_x = x + delta
            if "di" in self.attack_method:
                img_x = self.input_diversity(img_x)

            outputs = self.model(img_x)
            loss = self.loss_fn(outputs, y)
            loss.backward()

            grad = delta.grad.data

            # Gaussian kernel: TI-FGSM
            if "ti" in self.attack_method:
                # grad = F.conv2d(grad, kernel, padding=self.kernlen // 2)
                # grad = F.conv2d(grad, kernel, padding=(self.kernlen//2, self.kernlen//2), groups=3)
                grad = self.kernel_conv(grad, ti_kernel, kern_size=(self.kernlen//2, self.kernlen//2), groups=3)

            # momentum: MI-FGSM / NI-FGSM
            if "mi" in self.attack_method or "ni" in self.attack_method:
                # g = self.decay_factor * g + normalize_by_pnorm(grad, p=1)
                g = self.decay_factor * g + grad / torch.abs(grad).mean([1,2,3], keepdim=True)
                grad = g
                
            # Patch-wise attach: PI-FGSM
            amplification += eps_iter * grad.data.sign()
            cut_noise = torch.clamp(abs(amplification) - self.eps, 0, 1e5) * amplification.sign()
            projection = (
                self.pi_gamma
                * (self.kernel_conv(cut_noise, pi_kernel, self.pi_kern_size//2)).sign()
            )
            amplification += projection

            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign + projection
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x

            delta.grad.data.zero_()

        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        return x_adv


    def input_diversity(self, img, prob=0.5, rescale=None):
        size = img.size(2)
        if rescale is None:
            rescale = int(size / 0.875)

        gg = torch.rand(1).item()
        if gg >= prob:
            return img
        else:
            rnd = torch.randint(size, rescale + 1, (1,)).item()
            rescaled = F.interpolate(img, (rnd, rnd), mode="nearest")
            h_rem = rescale - rnd
            w_hem = rescale - rnd
            pad_top = torch.randint(0, h_rem + 1, (1,)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_hem + 1, (1,)).item()
            pad_right = w_hem - pad_left
            padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
            padded = F.interpolate(padded, (size, size), mode="nearest")
            return padded

    def get_Gaussian_kernel(self, kernlen=21, nsig=3):
        # define Gaussian kernel
        kern1d = st.norm.pdf(np.linspace(-nsig, nsig, kernlen))
        kernel = np.outer(kern1d, kern1d)
        kernel = kernel / kernel.sum()
        # kernel = torch.FloatTensor(kernel).expand(
        #     x.size(1), x.size(1), kernlen, kernlen
        # )
        # kernel = kernel.to(x.device)
        # return kernel
        stack_kern = np.stack([kernel, kernel, kernel])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.FloatTensor(stack_kern).cuda()
        # print(stack_kern.shape)
        return stack_kern

    def get_project_kernel(self, kern_size=3):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).cuda()
        return stack_kern

    def kernel_conv(self, x, stack_kern, kern_size, groups=3):
        x = F.conv2d(x, stack_kern, padding=(kern_size, kern_size), groups=groups)
        return x

