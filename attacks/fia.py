import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F

from attacks.utils import normalize_by_pnorm

from .base_attacker import Based_Attacker


layer_name_map = {
    "inceptionv3": "Mixed_5b",
    "inceptionresnetv2": "conv2d_4a",
    "vgg16": "_features.15",
    "resnet152": "layer4.2"
}
    
class FIA_Attacker(Based_Attacker):
    def get_config(arch):
        config = {
            "eps": 16,
            "nb_iter": 10,
            "eps_iter": 1.6,
            "prob": 0.5,  
            "kernlen": 7, 
            "nsig": 3,
            "decay_factor": 1.0,  

            "fia_ens": 30,                         # for FIA
            "fia_probb": 0.9,                      # for FIA (1 - 0.9 = 0.1)
            "fia_opt_layer": layer_name_map[arch], # for FIA (incep_v3)
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
            "scale_copies": 5,
            "vi_sample_n": 20,
            "vi_sample_beta": 1.5,
            "pi_beta": 10,
            "pi_gamma": 16,
            "pi_kern_size": 3,
            # for FIA
            "fia_ens": 30,
            "fia_probb": 0.9,
            "fia_opt_layer": "",
        }
        for k, v in default_value.items():
            self.load_params(k, v, args)
        
        super().__init__(
            attack_method=attack_method,
            model=model,
            loss_fn=loss_fn,
            args=args
        )
        
        self.mediate_grad   = None
        self.mediate_output = None
        

    def load_params(self, key, value, args):
        try:
            self.__dict__[key] = vars(args)[key]
        except:
            self.__dict__[key] = value

    def perturb(self, x, y):
        feature_weights = self.get_feature_weights(x, y)
        # return super().perturb(x, y)

        eps_iter = self.eps_iter

        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            img_x = x
            outputs = self.model(img_x + delta)

            loss = self.get_loss(feature_weights)
            if self.target:
                loss = -loss

            loss.backward()
            grad = delta.grad.data

            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x

            delta.grad.data.zero_()

        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        return x_adv


    def register_fw_hook(self, module, x_in, x_out):
        self.mediate_output = x_out.clone()

    def register_bp_hook(self, module, grad_in, grad_out):
        self.mediate_grad = torch.cat(grad_in, 1)

    def get_feature_weights(self, x, y):
        mediate_module = None
        for name, module in self.model.model.named_modules():
            if name == self.fia_opt_layer:
                mediate_module = module
                break
        assert mediate_module is not None, f"Can not find module named {self.fia_opt_layer} in {self.model.model_name}."

        mediate_module.register_forward_hook(self.register_fw_hook)
        mediate_module.register_backward_hook(self.register_bp_hook)

        bi_prob = torch.ones_like(x) * self.fia_probb
        weights = 0.
        for ens in range(self.fia_ens):
            if ens == 0:
                mask = torch.ones_like(x)
            else:   
                mask = torch.bernoulli(bi_prob)
            outputs = self.model(x * mask)

            loss = F.cross_entropy(outputs, y)
            loss.backward()

            weights += self.mediate_grad
        
        # import ipdb; ipdb.set_trace()
        weights = normalize_by_pnorm(weights, p=2)   
        return weights

    def get_loss(self, weights):
        # loss = 0.
        # for med_out in self.mediate_output:
        #     import ipdb; ipdb.set_trace()
        #     loss += (med_out*weights).sum() / med_out.view(-1).shape[0]
        # return loss / len(self.mediate_output)
        return (self.mediate_output*weights).sum() / self.mediate_output.view(-1).shape[0]
        
