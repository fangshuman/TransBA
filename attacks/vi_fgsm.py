import torch

from .basic import I_FGSM_Attack, MI_FGSM_Attack


class VI_FGSM_Attack(I_FGSM_Attack):
    def __init__(
        self,
        model,
        loss_fn,
        eps=0.05,
        nb_iter=10,
        eps_iter=0.005,
        sample_n=20,
        sample_beta=1.5,
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
        self.sample_n = sample_n
        self.sample_beta = sample_beta
        self.variance = None


    def grad_preprocess(self, x, delta, y):
        grad = super().grad_preprocess(x, delta, y)

        global_grad = torch.zeros_like(x)
        for i in range(self.sample_n):
            r = torch.rand_like(x) * self.sample_beta * self.eps
            r.requires_grad_()

            outputs = self.model(x + delta + r)
            loss = self.loss_fn(outputs, y)
            if self.target: 
                loss = -loss
            
            loss.backward()
            global_grad += r.grad.data
            r.grad.data.zero_()

        current_grad = grad + self.variance
        
        # update variance
        self.variance = global_grad / self.sample_n - grad

        return current_grad


    def init_extra_var(self, x):
        self.variance = torch.zeros_like(x)

