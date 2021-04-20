import ipdb
import numpy as np
import torch
import torch.nn as nn

from .base import Attack

class SGM_Attack(Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.gamma = gamma
        
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
        return super().perturb(x, y)




class SGM_Attack_for_ResNet(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
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
        '''
        vgg11: [*, M, *, M, *, *, M, *, *, M, *, *, M],
        vgg13: [*, *, M, *, *, M, *, *, M, *, *, M, *, *, M],
        vgg16: [*, *, M, *, *, M, *, *, *, M, *, *, *, M, *, *, *, M],
        vgg19: [*, *, M, *, *, M, *, *, *, *, M, *, *, *, *, M, *, *, *, *, M],

        * consists of "conv, relu" in vgg while consists of "conv, bn, relu" in vgg_bn    
        '''
        # Consider features between two nearest Maxpool as a module
        # There are 5 modules in vgg16_bn
        # and each module has 2, 2, 3, 3, 3 ReLU
        pos_relu =  [2, 5,   9, 12,   16, 19, 22,   26, 29, 32,   36, 39, 42]
        cof_gamma = [2, 2,   2,  2,    3,  3,  3,    3,  3,  3,    3,  3,  3]

        for name, module in self.model.named_modules():
            if 'features.' in name:
                feature_id = int(name.split('.')[-1])
                if feature_id in pos_relu:
                    coff = cof_gamma[pos_relu.index(feature_id)]
                    gamma = np.power(self.gamma, 1.0 / coff)
                    backward_hook_sgm = self.backward_hook(gamma)
                    module.register_backward_hook(backward_hook_sgm)



class SGM_Attack_for_InceptionV3(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        cof_gamma = {
            'Conv2d': 1,
            'Mixed_5b': 7, 'Mixed_5c': 7, 'Mixed_5d': 7,
            'Mixed_6a': 4, 'Mixed_6b': 10, 'Mixed_6c': 10, 'Mixed_6d': 10, 'Mixed_6e': 10,
            'AuxLogits': 2,
            'Mixed_7a': 6, 'Mixed_7b': 9, 'Mixed_7c': 9,
        }
        for name, module in self.model.named_modules():
            if 'relu' in name:
                if 'Conv2d' in name:
                    coff = 1
                else:
                    coff = cof_gamma[name.split('.')[0]]
                gamma = np.power(self.gamma, 1.0 / coff)
                backward_hook_sgm = self.backward_hook(gamma)
                module.register_backward_hook(backward_hook_sgm)



class SGM_Attack_for_InceptionV4(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        cof_gamma = [1, 1, 1, 1, 6, 1, 7, 7, 7, 7, 4, 10, 10, 10, 10, 10, 10, 10, 6, 10, 10, 10]
        for name, module in self.model.named_modules():
            if 'relu' in name:
                feature_id = int(name.split('.')[1])
                coff = cof_gamma[feature_id]
                gamma = np.power(self.gamma, 1.0 / coff)
                backward_hook_sgm = self.backward_hook(gamma)
                module.register_backward_hook(backward_hook_sgm)



class SGM_Attack_for_InceptionResNetV2(SGM_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, gamma=0.5, target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, gamma=gamma, target=target)

    def register_hook(self):
        cof_gamma = {
            'Conv2d': 1,
            'mixed_5b': 7, 
            'repeat': 7,
            'mixed_6a': 4, 
            'repeat_1': 5, 
            'mixed_7a': 7,
            'repeat_2': 5,
            'block8': 4,
            'conv2d_7b': 1,
        }
        for name, module in self.model.named_modules():
            if 'relu' in name:
                if 'Conv2d' in name:
                    coff = 1
                else:
                    coff = cof_gamma[name.split('.')[0]]
                gamma = np.power(self.gamma, 1.0 / coff)
                backward_hook_sgm = self.backward_hook(gamma)
                module.register_backward_hook(backward_hook_sgm)