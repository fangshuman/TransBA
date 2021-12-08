from operator import imod
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F

from .utils import normalize_by_pnorm
from .base import Attack

    
class IFGSM_Based_Attacker(Attack):
    def get_config(arch):
        config = {
            "eps": 16,
            "nb_iter": 10,
            "eps_iter": 1.6,
            "prob": 0.5,  
            "kernlen": 7, 
            "nsig": 3,
            "decay_factor": 1.0,  
            "scale_copies": 5,      # for si
            "vi_sample_n": 20,      # for vi
            "vi_sample_beta": 1.5,  # for vi
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
            "scale_copies": 5,
            "vi_sample_n": 20,
            "vi_sample_beta": 1.5,
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
            kernel = self.get_Gaussian_kernel(x, kernlen=self.kernlen, nsig=self.nsig)
        if "mi" in self.attack_method or "ni" in self.attack_method: 
            g = torch.zeros_like(x)
        if "vi" in self.attack_method:
            variance = torch.zeros_like(x)
        if "pi" in self.attack_method:
            amplification = torch.zeros_like(x)
            eps_iter *= self.pi_beta
            pi_kernel = self.get_project_kernel(kern_size=self.pi_kern_size)
            if self.pi_gamma > 1:
                self.pi_gamma /= 255.

        extra_item = torch.zeros_like(x)
        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            if "ni" in self.attack_method:
                img_x = x + self.decay_factor * eps_iter * g
            else:
                img_x = x

            # scale-invariant: SI-FGSM
            if "si" in self.attack_method:
                grad = torch.zeros_like(img_x)
                for i in range(self.scale_copies):
                    if "di" in self.attack_method:
                        outputs = self.model(
                            self.input_diversity((img_x + delta) * (1.0 / pow(2, i)), prob=self.prob)
                        )
                    else:
                        outputs = self.model((img_x + delta) * (1.0 / pow(2, i)))

                    loss = self.loss_fn(outputs, y)
                    if self.target:
                        loss = -loss

                    loss.backward()
                    grad += delta.grad.data
                    delta.grad.data.zero_()
                # get average value of gradient
                grad = grad / self.scale_copies

            # variance: VI-FGSM
            elif "vi" in self.attack_method:
                global_grad = torch.zeros_like(img_x)

                for i in range(self.vi_sample_n):
                    r = (torch.rand_like(img_x) - 0.5) * 2  # scale [0,1] to [-1,1]
                    r *= self.vi_sample_beta * self.eps  # scale [-1,1] to [-beta*eps, beta*eps]
                    r.requires_grad_()

                    if "di" in self.attack_method:
                        outputs = self.model(self.input_diversity((img_x + delta).data + r, prob=self.prob))
                    else:
                        outputs = self.model((img_x + delta).data + r)

                    loss = self.loss_fn(outputs, y)
                    if self.target:
                        loss = -loss

                    loss.backward()
                    global_grad += r.grad.data
                    r.grad.data.zero_()

                current_grad = grad + variance

                # update variance
                variance = global_grad / (1.0 * self.vi_sample_n) - grad

                # return current_grad
                grad = current_grad

            else:
                if "di" in self.attack_method:
                    outputs = self.model(self.input_diversity(img_x + delta, prob=self.prob))
                else:
                    outputs = self.model(img_x + delta)

                loss = self.loss_fn(outputs, y)
                if self.target:
                    loss = -loss

                loss.backward()
                grad = delta.grad.data

            # Gaussian kernel: TI-FGSM
            if "ti" in self.attack_method:
                grad = F.conv2d(grad, kernel, padding=self.kernlen // 2)

            # momentum: MI-FGSM / NI-FGSM
            if "mi" in self.attack_method or "ni" in self.attack_method:
                # g = self.decay_factor * g + normalize_by_pnorm(grad, p=1)
                # g = self.decay_factor * g + grad / torch.abs(grad).sum([1,2,3], keepdim=True)
                g = self.decay_factor * g + grad / torch.abs(grad).mean([1,2,3], keepdim=True)
                grad = g
                
            # Patch-wise attach: PI-FGSM
            if "pi" in self.attack_method:
                amplification += eps_iter * grad.data.sign()
                cut_noise = torch.clamp(abs(amplification) - self.eps, 0, 1e5) * amplification.sign()
                projection = (
                    self.pi_gamma
                    * (self.project_noise(cut_noise, pi_kernel, self.pi_kern_size//2)).sign()
                )
                amplification += projection
                extra_item = projection  # return extra item

            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign + extra_item
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x

            delta.grad.data.zero_()

        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        return x_adv


    def input_diversity(self, img, prob=0.5):
        size = img.size(2)
        resize = int(size / 0.875)

        gg = torch.rand(1).item()
        if gg >= prob:
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

    def get_Gaussian_kernel(self, x, kernlen=21, nsig=3):
        # define Gaussian kernel
        kern1d = st.norm.pdf(np.linspace(-nsig, nsig, kernlen))
        kernel = np.outer(kern1d, kern1d)
        kernel = kernel / kernel.sum()
        kernel = torch.FloatTensor(kernel).expand(
            x.size(1), x.size(1), kernlen, kernlen
        )
        kernel = kernel.to(x.device)
        return kernel

    def get_project_kernel(self, kern_size=3):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).cuda()
        return stack_kern

    def project_noise(self, x, stack_kern, kern_size):
        x = F.conv2d(x, stack_kern, padding=(kern_size, kern_size), groups=3)
        return x

