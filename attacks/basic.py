import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as st

from .base import Attack


class I_FGSM_Attack(Attack):
    def __init__(
        self,
        model,
        loss_fn,
        eps=0.05,
        nb_iter=10,
        eps_iter=0.005,
        target=False,
    ):
        super(I_FGSM_Attack, self).__init__(model)
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.target = target

    def preprocess(self, x, delta, y):
        return delta

    def postprocess(self, x, delta, y):
        pass

    def grad_postprocess(self, grad):
        return grad.sign()

    def perturb_one_iter(self, x, delta, y):
        outputs = self.model(x + delta)
        loss = self.loss_fn(outputs, y)
        if self.target:
            loss = -loss

        loss.backward()

        # TODO: L_inf only now.
        grad_sign = self.grad_postprocess(delta.grad.data)
        delta.data = delta.data + self.eps_iter * grad_sign
        delta.data = torch.clamp(delta.data, -self.eps, self.eps)
        delta.data = torch.clamp(x.data + delta, 0.0, 1.0) - x

        delta.grad.data.zero_()
        return delta

    def perturb(self, x, y):
        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            delta = self.preprocess(x, delta, y)

            delta = self.perturb_one_iter(x, delta, y)
            delta.requires_grad_(True)

            self.postprocess(x, delta, y)

        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        return x_adv


class TI_FGSM_Attack(I_FGSM_Attack):
    def __init__(
        self,
        model,
        loss_fn,
        eps=0.05,
        nb_iter=10,
        eps_iter=0.005,
        kernlen=7,
        nsig=3,
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
        self.kernlen = kernlen
        self.nsig = nsig
        # define Gaussian kernel
        kern1d = st.norm.pdf(np.linspace(-self.nsig, self.nsig, self.kernlen))
        kernel = np.outer(kern1d, kern1d)
        kernel = kernel / kernel.sum()
        self._kernl = torch.FloatTensor(kernel).to(next(model.parameters()).device)

    def grad_postprocess(self, grad):
        kernel = self.kernel.expand(
            grad.size(1), grad.size(1), self.kernlen, self.kernlen
        )
        grad_sign = (F.conv2d(grad, kernel, padding=self.kernlen // 2)).sign()
        return grad_sign


class DI_FGSM_Attack(I_FGSM_Attack):
    def __init__(
        self,
        model,
        loss_fn,
        eps=0.05,
        nb_iter=10,
        eps_iter=0.005,
        prob=0.5,
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
        self.prob = prob

    def input_diversity(self, img):
        size = img.size(2)
        resize = int(size / 0.875)

        gg = torch.rand(1).item()
        if gg >= self.prob:
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

    def preprocess(self, x, delta, y):
        x_d = self.input_diversity(x + delta)
        d = (x_d) - x
        d.requires_grad_(True)
        return d


class MI_FGSM_Attack(I_FGSM_Attack):
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

    def normalize_by_pnorm(x, small_constant=1e-6):
        batch_size = x.size(0)
        norm = x.abs().view(batch_size, -1).sum(dim=1)
        norm = torch.max(norm, torch.ones_like(norm) * small_constant)
        return (x.transpose(0, -1) * (1.0 / norm)).transpose(0, -1).contiguous()

    def grad_postprocess(self, grad):
        self.g = self.decay_factor * self.g + MI_FGSM_Attack.normalize_by_pnorm(grad)
        return self.g.sign()

    def perturb(self, x, y):
        self.g = torch.zeros_like(x)
        return super().perturb(x, y)

