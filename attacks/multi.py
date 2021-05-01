from .basic import *
from .ni_si_fgsm import *
from .vi_fgsm import *

class Multi_I_FGSM_Attack(I_FGSM_Attack):
    def __init__(
        self,
        attack_method,
        model,
        loss_fn,
        args,
    ):
        super().__init__(
            model,
            loss_fn,
            eps=args.eps,
            nb_iter=args.nb_iter,
            eps_iter=args.eps_iter,
            target=args.target,
        )
        
        # get instance
        self.method_list = {
            # 'ti_fgsm': None,
            # 'di_fgsm': None,
            # 'mi_fgsm': None,
            # 'ni_fgsm': None,
            # 'si_fgsm': None,
            # 'vi_fgsm': None,
        } 
        for m in attack_method.split("_")[:-1]:
            # self.method_list.append(get_attack(m))
            if m == "i":
                pass
            elif m == "di":
                self.method_list["di_fgsm"] = DI_FGSM_Attack(
                    self.model, 
                    self.loss_fn,
                    eps=args.eps,
                    nb_iter=args.nb_iter,
                    eps_iter=args.eps_iter,
                    prob=args.prob,
                    target=args.target
                    )
            elif m == "ti":
                self.method_list["ti_fgsm"] = TI_FGSM_Attack(
                    self.model, 
                    self.loss_fn,
                    eps=args.eps,
                    nb_iter=args.nb_iter,
                    eps_iter=args.eps_iter,
                    kernlen=args.kernlen,
                    nsig=args.nsig,
                    target=args.target
                    )
            elif m == "mi":
                self.method_list["mi_fgsm"] = MI_FGSM_Attack(
                    self.model, 
                    self.loss_fn,
                    eps=args.eps,
                    nb_iter=args.nb_iter,
                    eps_iter=args.eps_iter,
                    decay_factor=args.decay_factor,
                    target=args.target
                    )
            elif m == "ni":
                self.method_list["ni_fgsm"] = NI_FGSM_Attack(
                    self.model, 
                    self.loss_fn,
                    eps=args.eps,
                    nb_iter=args.nb_iter,
                    eps_iter=args.eps_iter,
                    decay_factor=args.decay_factor,
                    target=args.target
                    )
            elif m == "si":
                self.method_list["si_fgsm"] = SI_FGSM_Attack(
                    self.model, 
                    self.loss_fn,
                    eps=args.eps,
                    nb_iter=args.nb_iter,
                    eps_iter=args.eps_iter,
                    scale_copies=args.scale_copies,
                    target=args.target
                    )
            elif m == "vi":
                self.method_list["vi_fgsm"] = VI_FGSM_Attack(
                    self.model, 
                    self.loss_fn,
                    eps=args.eps,
                    nb_iter=args.nb_iter,
                    eps_iter=args.eps_iter,
                    sample_n=args.vi_sample_n,
                    sample_beta=args.vi_sample_beta,
                    target=args.target
                    )
            else:
                raise NotImplementedError("Current code only supports ti/di/mi/ni/si/vi Please check attack method name.")

    def init_extra_var(self, x):
        for m in self.method_list:
            self.method_list[m].init_extra_var(x)

    def preprocess(self, x, delta, y):
        for m in self.method_list:
            x, delta = self.method_list[m].preprocess(x, delta, y)
        return x, delta

    def postprocess(self, x, delta, y):
        for m in self.method_list:
            self.method_list[m].postprocess(x, delta, y)

    def grad_preprocess(self, x, delta, y):
        grad = super().grad_preprocess(x, delta, y)
        for m in self.method_list:
            grad = self.method_list[m].grad_preprocess(x, delta, y)
        return grad

    def grad_processing(self, grad):
        for m in self.method_list:
            grad = self.method_list[m].grad_processing(grad)
        return grad

    def grad_postprocess(self, grad):
        grad_sign = grad.sign()
        for m in self.method_list:
            grad_sign = self.method_list[m].grad_postprocess(grad)
        return grad_sign


    # def perturb(self, x, y):
    #     import ipdb; ipdb.set_trace()
    #     return super().perturb(x, y)