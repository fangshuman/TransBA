import ipdb
import numpy as np
import torch
import torch.nn as nn

from .AWP import AdvWeightPerturb

class SGM_Attack(object):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False, awp=False):
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.target = target
        self.gamma = gamma
        if awp:
            self.awp = AdvWeightPerturb(model)
        else:
            self.awp = None
        
    def backward_hook(self, gamma):
        # implement SGM through grad through ReLU
        def _backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (gamma * grad_in[0],)
        return _backward_hook

    def backward_hook_norm(self, module, grad_in, grad_out):
        # normalize the gradient to avoid gradient explosion or vanish
        std = torch.std(grad_in[0])
        return (grad_in[0] / std,)

    def perturb(self, x, y):
        delta = torch.zeros_like(x)
        delta.requires_grad_()
    
        for i in range(self.nb_iter):
            # awp
            if self.awp is not None:
                diff = self.awp.calc_awp(x + delta, y)
                self.awp.perturb(diff)
                delta.grad.data.zero_()
                
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

            if self.awp is not None:
                self.awp.restore(diff)

        x_adv = torch.clamp(x + delta, 0., 1.)
        return x_adv



class SGM_Attack_for_ResNet(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.2, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        # only use ResNet-50
        # There are 2 ReLU in Conv module ResNet-50
        gamma = np.power(self.gamma, 0.5)
        backward_hook_sgm = self.backward_hook(gamma)

        for name, module in self.model.named_modules():
            if 'relu' in name and not '0.relu' in name:
                module.register_backward_hook(backward_hook_sgm)
            if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
                module.register_backward_hook(self.backward_hook_norm)



class SGM_Attack_for_DenseNet(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        # There are 2 ReLU in Conv module DenseNet-121
        gamma = np.power(self.gamma, 0.5)
        backward_hook_sgm = self.backward_hook(gamma)

        for name, module in self.model.named_modules():
            if 'relu' in name and not 'transition' in name:
                module.register_backward_hook(backward_hook_sgm)



class SGM_Attack_for_VGG(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        pos_relu =  [2, 5,   9, 12,   16, 19, 22,   26, 29, 32,   36, 39, 42]
        
        gamma = self.gamma
        backward_hook_sgm = self.backward_hook(gamma)
        for p in pos_relu:
            self.model._features[p].register_backward_hook(backward_hook_sgm)



class SGM_Attack_for_InceptionV3(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        module_names = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c']
        
        gamma = self.gamma
        backward_hook_sgm = self.backward_hook(gamma)
        for mm in module_names:
            self.model._modules.get('mm')._modules.get('relu').register_backward_hook(backward_hook_sgm)

        # cof_gamma = {
        #     'Conv2d': 1,
        #     'Mixed_5b': 7, 'Mixed_5c': 7, 'Mixed_5d': 7,
        #     'Mixed_6a': 4, 'Mixed_6b': 10, 'Mixed_6c': 10, 'Mixed_6d': 10, 'Mixed_6e': 10,
        #     'AuxLogits': 2,
        #     'Mixed_7a': 6, 'Mixed_7b': 9, 'Mixed_7c': 9,
        # }
        # for name, module in self.model.named_modules():
        #     if 'relu' in name:
        #         # if 'Conv2d' in name:
        #         #     coff = 1
        #         # else:
        #         #     coff = cof_gamma[name.split('.')[0]]
        #         # gamma = np.power(self.gamma, 1.0 / coff)
        #         gamma = self.gamma
        #         backward_hook_sgm = self.backward_hook(gamma)
        #         module.register_backward_hook(backward_hook_sgm)



class SGM_Attack_for_InceptionV4(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        gamma = self.gamma
        backward_hook_sgm = self.backward_hook(gamma)
        for f in range(22):  # 21 modules
            self.model.features[f]._modules.get('relu').register_backward_hook(backward_hook_sgm)
        
        # cof_gamma = [1, 1, 1, 1, 6, 1, 7, 7, 7, 7, 4, 10, 10, 10, 10, 10, 10, 10, 6, 10, 10, 10]
        # for name, module in self.model.named_modules():
        #     if 'relu' in name:
        #         # feature_id = int(name.split('.')[1])
        #         # coff = cof_gamma[feature_id]
        #         # gamma = np.power(self.gamma, 1.0 / coff)
        #         gamma = self.gamma
        #         backward_hook_sgm = self.backward_hook(gamma)
        #         module.register_backward_hook(backward_hook_sgm)


class SGM_Attack_for_InceptionResNetV2(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        module_names = ['conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'maxpool_3a', 'conv2d_3b', 'conv2d_4a', 'maxpool_5a', 'mixed_5b', 'repeat', 'mixed_6a','repeat_1', 'mixed_7a', 'repeat_2', 'block8', 'conv2d_7b', 'avgpool_1a', 'last_linear']

        gamma = self.gamma
        backward_hook_sgm = self.backward_hook(gamma)
        for mm in module_names:
            self.model._modules.get('mm')._modules.get('relu').register_backward_hook(backward_hook_sgm)

        # cof_gamma = {
        #     'Conv2d': 1,
        #     'mixed_5b': 7, 
        #     'repeat': 7,
        #     'mixed_6a': 4, 
        #     'repeat_1': 5, 
        #     'mixed_7a': 7,
        #     'repeat_2': 5,
        #     'block8': 4,
        #     'conv2d_7b': 1,
        # }
        # for name, module in self.model.named_modules():
        #     if 'relu' in name:
        #         # if 'Conv2d' in name:
        #         #     coff = 1
        #         # else:
        #         #     coff = cof_gamma[name.split('.')[0]]
        #         # gamma = np.power(self.gamma, 1.0 / coff)
        #         gamma = self.gamma
        #         backward_hook_sgm = self.backward_hook(gamma)
        #         module.register_backward_hook(backward_hook_sgm)