import torch

from .basic import I_FGSM_Attack
from .utils import normalize_by_pnorm


class NI_FGSM_Attack(I_FGSM_Attack):
    def __init__(
        self,
        model,
        loss_fn,
        eps=0.05,
        nb_iter=10,
        eps_iter=0.005,
        decay_factor=1.0,
        target=False,
    ):
        super().__init__(
            model,
            loss_fn,
            eps=eps,
            nb_iter=nb_iter,
            eps_iter=eps_iter,
            target=target,
        )
        self.decay_factor = decay_factor
        self.g = None

    # def normalize_by_pnorm(x, small_constant=1e-6):
    #     batch_size = x.size(0)
    #     norm = x.abs().view(batch_size, -1).sum(dim=1)
    #     norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    #     return (x.transpose(0, -1) * (1.0 / norm)).transpose(0, -1).contiguous()

    # def preprocess(self, x, delta, y):
    #     x_nes = (x + delta) + self.decay_factor * self.eps_iter * self.g
    #     return x_nes, delta
    

    def grad_processing(self, grad):
        self.g = self.decay_factor * self.g + normalize_by_pnorm(grad, p=1)
        return self.g

    def init_extra_var(self, x):
        self.g = torch.zeros_like(x)

    def perturb_one_iter(self, x, delta, y):
        x_nes = (x + delta) + self.decay_factor * self.eps_iter * self.g
        grad = self.grad_preprocess(x_nes, delta, y)

        # TODO: L_inf only now.
        grad = self.grad_processing(grad)
        grad_sign = self.grad_postprocess(grad)

        delta.data = delta.data + self.eps_iter * grad_sign
        delta.data = torch.clamp(delta.data, -self.eps, self.eps)
        delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x

        delta.grad.data.zero_()
        return delta



class SI_FGSM_Attack(I_FGSM_Attack):
    def __init__(
        self,
        model,
        loss_fn,
        eps=0.05,
        nb_iter=10,
        eps_iter=0.005,
        scale_copies=5,  
        target=False,
    ):
        super().__init__(
            model,
            loss_fn,
            eps=eps,
            nb_iter=nb_iter,
            eps_iter=eps_iter,
            target=target,
        )
        self.scale_copies = scale_copies

    def grad_preprocess(self, x, delta, y):
        grad = torch.zeros_like(x)
        for i in range(self.scale_copies):
            outputs = self.model((x  + delta) * (1. / pow(2,i)))
            loss = self.loss_fn(outputs, y)
            if self.target:
                loss = -loss

            loss.backward()
            grad += delta.grad.data
            delta.grad.data.zero_()

        return grad / self.scale_copies
        # return grad


