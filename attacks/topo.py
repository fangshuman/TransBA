import torch

from .utils import normalize_by_pnorm
from .base import Attack


def quan(img):
    img = torch.clamp(img, 0.0, 1.0)
    # return img
    img = img * 255.0
    img = torch.round(img)
    return img / 255.0


class Topo(Attack):
    config = {
        "eps": 16,
        "nb_iter": 10,
        "eps_iter": 1.6,
        "gamma": 1.0,  # using sgm when gamma < 1.0
        "prob": 0.5,  # for di
        "kernlen": 7,  # for ti
        "nsig": 3,  # for ti
        "decay_factor": 1.0,  # for mi/ni
        "scale_copies": 5,  # for si
        "vi_sample_n": 20,  # for vi
        "vi_sample_beta": 1.5,  # for vi
        "amplification": 10,  # for pi (patch-wise)
    }

    def __init__(self, attack_method, model, loss_fn, args):
        self.model = model
        self.loss_fn = loss_fn
        self.args = args
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
            "sample_n": 20,
            "sample_beta": 1.5,
            "amplification": 10,
        }
        for k, v in default_value.items():
            self.load_params(k, v, args)

    def load_params(self, key, value, args):
        try:
            self.__dict__[key] = vars(args)[key]
        except:
            self.__dict__[key] = value

    def perturb(self, x, y):
        eps_iter = self.eps_iter
        g = torch.zeros_like(x)

        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            img_x = x

            outputs = self.model(img_x + delta)
            loss = self.loss_fn(outputs, y)
            loss.backward()

            grad = delta.grad.data

            g = self.decay_factor * g + normalize_by_pnorm(grad, p=1)
            grad = g

            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            # delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x
            delta.data = quan(x.data + delta) - x

            delta.grad.data.zero_()

        # x_adv = torch.clamp(x + delta, 0.0, 1.0)
        x_adv = quan(x + delta)
        return x_adv
