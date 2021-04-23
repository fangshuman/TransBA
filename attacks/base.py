import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st

from dataset import save_image


class I_FGSM_Attack(object):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, target=False):
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.target = target

    def perturb(self, x, y):
    #def perturb(self, x, y, indexs, img_list):
        delta = torch.zeros_like(x)
        delta.requires_grad_()
    
        for i in range(self.nb_iter):
            outputs = self.model(x + delta)
            loss = self.loss_fn(outputs, y)
            if self.target:
                loss = -loss
        
            loss.backward()

            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + self.eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0., 1.) - x

            delta.grad.data.zero_()

            # save
            # if (i + 1) % 10 == 0:
            #     print(f'saving {i + 1} iter adversarial examples for this batch.')
            #     output_dir = 'output/ifgsm_resnet50/' + str(i + 1) + 'iter'
            #     if not os.path.exists(output_dir):
            #         os.mkdir(output_dir)
            #     imgs = torch.clamp(x + delta, 0., 1.)
            #     save_image(imgs.detach().cpu().numpy(), indexs, img_list, output_dir)

        x_adv = torch.clamp(x + delta, 0., 1.)
        return x_adv



class TI_FGSM_Attack(I_FGSM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, kernlen=7, nsig=3, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.kernlen = kernlen
        self.nsig = nsig

    def perturb(self, x, y):
        # define Gaussian kernel
        kern1d = st.norm.pdf(np.linspace(-self.nsig, self.nsig, self.kernlen))
        kernel = np.outer(kern1d, kern1d)
        kernel = kernel / kernel.sum()
        kernel = torch.FloatTensor(kernel).expand(x.size(1), x.size(1), self.kernlen, self.kernlen)
        kernel = kernel.to(x.device)
    
        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            outputs = self.model(x + delta)
            loss = self.loss_fn(outputs, y)
            if self.target:
                loss = -loss
        
            loss.backward()

            grad_sign = (F.conv2d(delta.grad.data, kernel, padding=self.kernlen//2)).sign()
            delta.data = delta.data + self.eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0., 1.) - x

            delta.grad.data.zero_()
    
        x_adv = torch.clamp(x + delta, 0., 1.)
        return x_adv



class DI_FGSM_Attack(I_FGSM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, prob=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.prob = prob

    def perturb(self, x, y):
        def input_diversity(img):
            size = x.size(2)
            resize = int(size / 0.875)

            gg = torch.rand(1).item()
            if gg >= self.prob:
                return img
            else:
                rnd = torch.randint(size, resize + 1, (1,)).item()
                rescaled = F.interpolate(img, (rnd, rnd), mode='nearest')
                h_rem = resize - rnd
                w_hem = resize - rnd
                pad_top = torch.randint(0, h_rem + 1, (1,)).item()
                pad_bottom = h_rem - pad_top
                pad_left = torch.randint(0, w_hem + 1, (1,)).item()
                pad_right = w_hem - pad_left
                padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
                padded = F.interpolate(padded, (size, size), mode='nearest')
                return padded

        delta = torch.zeros_like(x)
        delta.requires_grad_()
    
        for i in range(self.nb_iter):
            outputs = self.model(input_diversity(x + delta))
            loss = self.loss_fn(outputs, y)
            if self.target:
                loss = -loss
        
            loss.backward()

            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + self.eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0., 1.) - x

            delta.grad.data.zero_()
    
        x_adv = torch.clamp(x + delta, 0., 1.)
        return x_adv



class MI_FGSM_Attack(I_FGSM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, decay_factor=1.0, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.decay_factor = decay_factor

    def perturb(self, x, y):
    #def perturb(self, x, y, indexs, img_list):
        def normalize_by_pnorm(x, small_constant=1e-6):
            batch_size = x.size(0)
            norm = x.abs().view(batch_size, -1).sum(dim=1)
            norm = torch.max(norm, torch.ones_like(norm) * small_constant)
            return (x.transpose(0, -1) * (1. / norm)).transpose(0, -1).contiguous()
        
        delta = torch.zeros_like(x)
        delta.requires_grad_()

        g = torch.zeros_like(x)
    
        for i in range(self.nb_iter):
            outputs = self.model(x + delta)
            loss = self.loss_fn(outputs, y)
            if self.target:
                loss = -loss
        
            loss.backward()

            g = self.decay_factor * g + normalize_by_pnorm(delta.grad)
            #g = self.decay_factor * g + g / torch.norm(delta.grad, p=1, dim=(1, 2, 3), keepdim=True)

            g_sign = torch.sign(g)
            delta.data = delta.data + self.eps_iter * g_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0., 1.) - x

            delta.grad.data.zero_()

            # save
            # if (i + 1) % 10 == 0:
            #     print(f'saving {i + 1} iter adversarial examples for this batch.')
            #     output_dir = 'output/mifgsm_resnet50/' + str(i + 1) + 'iter'
            #     if not os.path.exists(output_dir):
            #         os.mkdir(output_dir)
            #     imgs = torch.clamp(x + delta, 0., 1.)
            #     save_image(imgs.detach().cpu().numpy(), indexs, img_list, output_dir)
    
        x_adv = torch.clamp(x + delta, 0., 1.)
        return x_adv






