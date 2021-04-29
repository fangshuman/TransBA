import ipdb
import numpy as np
import torch
import torch.nn as nn


def backward_hook(gamma):
    # implement SGM through grad through ReLU
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook

def backward_hook_norm(module, grad_in, grad_out):
    # normalize the gradient to avoid gradient explosion or vanish
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


class SGM_Attack(object):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.target = target
        self.gamma = gamma
    
    def perturb(self, x, y):
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

        x_adv = torch.clamp(x + delta, 0., 1.)
        return x_adv



class SGM_Attack_for_ResNet(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.2, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        # only use ResNet-50
        # There are 2 ReLU in Conv module ResNet-50
        gamma = np.power(self.gamma, 0.5)
        backward_hook_sgm = backward_hook(gamma)

        for name, module in self.model.named_modules():
            if 'relu' in name and not '0.relu' in name:
                module.register_backward_hook(backward_hook_sgm)
            if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
                module.register_backward_hook(backward_hook_norm)



class SGM_Attack_for_DenseNet(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        # There are 2 ReLU in Conv module DenseNet-121
        gamma = np.power(self.gamma, 0.5)
        backward_hook_sgm = backward_hook(gamma)

        for name, module in self.model.named_modules():
            if 'relu' in name and not 'transition' in name:
                module.register_backward_hook(backward_hook_sgm)

