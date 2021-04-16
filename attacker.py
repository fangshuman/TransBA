import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st

from utils import *


class Attacker(object):
    """
    Args:
        predict: forward pass function.
        loss_fn: loss function.
        eps: maximum distortion.
        nb_iter: number of iterations.
        eps_iter: attack step size.
        target: if the attack is targeted.
    """
    def __init__(self, attack, predict, loss_fn, eps, nb_iter, eps_iter, target=False):
        self.attack = attack
        self.predict = predict
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.target = target

    def perturb(self, x, y):
        """
        Args:
            x: input data.
            y: input labels.
        Return: 
            tensor containing the perturbed input.
        """
        if self.attack == 'i-fgsm':
            return i_fgsm_attack(x, y, 
                                 self.predict, 
                                 self.loss_fn,
                                 self.eps, self.nb_iter, self.eps_iter, 
                                 target=self.target)
        elif self.attack == 'ti-fgsm':
            return ti_fgsm_attack(x, y, 
                                  self.predict, 
                                  self.loss_fn,
                                  self.eps, self.nb_iter, self.eps_iter, 
                                  kernlen=7, nsig=3,
                                  target=self.target)
        elif self.attack == 'di-fgsm':
            pass
        elif self.attack == 'mi-fgsm':
            return mi_fgsm_attack(x, y,
                                  self.predict,
                                  self.loss_fn,
                                  self.eps, self.nb_iter, self.eps_iter,
                                  decay_factor=1.0,
                                  target=self.target)
        elif self.attack == 'TAP':
            pass
        elif self.attack == 'Ghost':
            pass
        elif self.attack == 'SGM':
            pass
        elif self.attack == 'LinBP':
            pass
        else:
            raise ValueError(f'Unknown attacker {self.attack}')


def i_fgsm_attack(x, y, predict, loss_fn, eps, nb_iter, eps_iter, target=False):
    delta = torch.zeros_like(x)
    delta = nn.Parameter(delta)
    delta.requires_grad_()
    
    for i in range(nb_iter):
        outputs = predict(x + delta)
        loss = loss_fn(outputs, y)
        if target:
            loss = -loss
        
        loss.backward()

        grad_sign = delta.grad.data.sign()
        delta.data = delta.data + eps_iter * grad_sign
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(x.data + delta, 0., 1.) - x

        delta.grad.data.zero_()
    
    x_adv = torch.clamp(x + delta, 0., 1.)
    return x_adv


def ti_fgsm_attack(x, y, predict, loss_fn, eps, nb_iter, eps_iter, kernlen=7, nsig=3, target=False):
    # 关于高斯核的参数, 论文里用的kernlen=15, nsig应该是10
    # 但是别的方法(ni, emi...)把ti作为baseline的时候用的是kernlen=7, nsig=3
    
    # define Gaussian kernel
    kern1d = st.norm.pdf(np.linspace(-nsig, nsig, kernlen))
    kernel = np.outer(kern1d, kern1d)
    kernel = kernel / kernel.sum()
    kernel = torch.FloatTensor(kernel).expand(x.size(1), x.size(1), kernlen, kernlen)
    kernel = kernel.to(x.device)
    
    delta = torch.zeros_like(x)
    delta = nn.Parameter(delta)
    delta.requires_grad_()

    for i in range(nb_iter):
        outputs = predict(x + delta)
        loss = loss_fn(outputs, y)
        if target:
            loss = -loss
        
        loss.backward()
        #import ipdb;ipdb.set_trace()
        grad_sign = (F.conv2d(delta.grad.data, kernel, padding=kernlen//2)).sign()
        delta.data = delta.data + eps_iter * grad_sign
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(x.data + delta, 0., 1.) - x

        delta.grad.data.zero_()
    
    x_adv = torch.clamp(x + delta, 0., 1.)
    return x_adv


def mi_fgsm_attack(x, y, predict, loss_fn, eps, nb_iter, eps_iter, decay_factor=1.0, target=False):
    delta = torch.zeros_like(x)
    delta = nn.Parameter(delta)
    delta.requires_grad_()

    g = torch.zeros_like(x)
    
    for i in range(nb_iter):
        outputs = predict(x + delta)
        loss = loss_fn(outputs, y)
        if target:
            loss = -loss
        
        loss.backward()

        g = decay_factor * g + normalize_by_pnorm(delta.grad, p=1)

        g_sign = torch.sign(g)
        delta.data = delta.data + eps_iter * g_sign
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(x.data + delta, 0., 1.) - x

        delta.grad.data.zero_()
    
    x_adv = torch.clamp(x + delta, 0., 1.)
    return x_adv


def SGM(x, y, predict, loss_fn, eps, nb_iter, eps_iter, gamma=0.5, target=False):
    def back_hook_sgm(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    
    gamma = np.power(gamma, 0.5)
    #for name, module in predict.named_modules():
