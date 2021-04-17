import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st


class Attack(object):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, target=False):
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.target = target

    def perturb(self, x, y):
        delta = torch.zeros_like(x)
        delta.requires_grad_()
    
        for i in range(self.nb_iter):
            outputs = self.model(x + delta)
            loss = loss_fn(outputs, y)
            if target:
                loss = -loss
        
            loss.backward()

            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + self.eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0., 1.) - x

            delta.grad.data.zero_()
    
        x_adv = torch.clamp(x + delta, 0., 1.)
        return x_adv



class I_FGSM_Attack(Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)



class TI_FGSM_Attack(Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, kerlen=7, nsig=3, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.kernlen = kerlen
        self.nsig = nsig

    def perturb(self, x, y):
        # define Gaussian kernel
        kern1d = st.norm.pdf(np.linspace(-nsig, nsig, kernlen))
        kernel = np.outer(kern1d, kern1d)
        kernel = kernel / kernel.sum()
        kernel = torch.FloatTensor(kernel).expand(x.size(1), x.size(1), kernlen, kernlen)
        kernel = kernel.to(x.device)
    
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            outputs = self.model(x + delta)
            loss = self.loss_fn(outputs, y)
            if self.target:
                loss = -loss
        
            loss.backward()

            grad_sign = (F.conv2d(delta.grad.data, kernel, padding=kernlen//2)).sign()
            delta.data = delta.data + self.eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0., 1.) - x

            delta.grad.data.zero_()
    
        x_adv = torch.clamp(x + delta, 0., 1.)
        return x_adv



class MI_FGSM_Attack(Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, decay_factor=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.decay_factor = decay_factor

    def perturb(self, x, y):
        def normalize_by_pnorm(x, p=2, small_constant=1e-6):
            assert isinstance(p, float) or isinstance(p, int)
            batch_size = x.size(0)
            norm = x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)
            norm = torch.max(norm, torch.ones_like(norm) * small_constant)
            return (x.transpose(0, -1) * (1. / norm)).transpose(0, -1).contiguous()
        
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        g = torch.zeros_like(x)
    
        for i in range(self.nb_iter):
            outputs = self.model(x + delta)
            loss = self.loss_fn(outputs, y)
            if self.target:
                loss = -loss
        
            loss.backward()

            g = decay_factor * g + normalize_by_pnorm(delta.grad, p=1)

            g_sign = torch.sign(g)
            delta.data = delta.data + self.eps_iter * g_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0., 1.) - x

            delta.grad.data.zero_()
    
        x_adv = torch.clamp(x + delta, 0., 1.)
        return x_adv





