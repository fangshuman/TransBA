from .basic import *
from .ni_si_fgsm import *
from .vi_fgsm import *
from .utils import normalize_by_pnorm

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
        self.method_list = {} 
        for m in attack_method.split("_")[:-1]:
            # self.method_list.append(get_attack(m))
            if m == "i":
                self.method_list["i_fgsm"] = I_FGSM_Attack(
                    self.model, 
                    self.loss_fn,
                    eps=args.eps,
                    nb_iter=args.nb_iter,
                    eps_iter=args.eps_iter,
                    target=args.target
                    )
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

    def get_output_x(self, x, delta, y):
        for m in self.method_list:
            return self.method_list[m].get_output_x(x, delta, y)

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

    def perturb_one_iter(self, x, delta, y):
        for m in self.method_list:
            delta = self.method_list[m].perturb_one_iter(x, delta, y)
        return delta




    # def perturb(self, x, y):
    #     import ipdb; ipdb.set_trace()
    #     return super().perturb(x, y)


class Multi_Attack(object):
    def __init__(
        self,
        attack_method,
        model,
        loss_fn,
        args,
    ):
        self.attack_method = attack_method
        self.model = model
        self.loss_fn = loss_fn

        try:
            # basic
            self.eps = args.eps
            self.nb_iter = args.nb_iter
            self.eps_iter = args.eps_iter
            self.target = args.target
            # extra
            self.prob = args.prob
            self.kernlen = args.kernlen
            self.nsig = args.nsig
            self.decay_factor = args.decay_factor
            self.scale_copies = args.scale_copies
            self.sample_n = args.vi_sample_n
            self.sample_beta = args.vi_sample_beta

        except:
            # basic default value
            self.eps = 0.05
            self.nb_iter = 10
            self.eps_iter = 0.005
            self.target = False
            # extra default value
            self.prob = 0.5
            self.kernlen = 7
            self.nsig = 3
            self.decay_factor = 1.0
            self.scale_copies = 5
            self.sample_n = 20
            self.sample_beta = 1.5



    def perturb(self, x, y):
        # initialize extra var
        if "mi" in self.attack_method or "ni" in self.attack_method:
            g = torch.zeros_like(x)
        if "vi" in self.attack_method:
            variance = torch.zeros_like(x)

        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            if "ni" in self.attack_method: 
                # img_x = (x + delta) + self.decay_factor * self.eps_iter * g
                img_x = x + self.decay_factor * self.eps_iter * g 
            else:
                img_x = x
            
            # get gradient
            grad = torch.zeros_like(img_x)
            if "si" in self.attack_method:
                for i in range(self.scale_copies):
                    if "di" in self.attack_method:
                        outputs = self.model(self.input_diversity(img_x + delta) * (1./pow(2,i)))
                    else:
                        outputs = self.model((img_x + delta) * (1./pow(2,i)))
                    
                    loss = self.loss_fn(outputs, y)
                    if self.target:
                        loss = -loss

                    loss.backward()
                    grad += delta.grad.data
                    delta.grad.data.zero_()

            else:
                if "di" in self.attack_method:
                    outputs = self.model(self.input_diversity(img_x + delta))
                # elif "ni" in self.attack_method:
                #     outputs = self.model(img_x)
                else:
                    outputs = self.model(img_x + delta)
                
                loss = self.loss_fn(outputs, y)
                if self.target:
                    loss = -loss
            
                loss.backward()
                grad = delta.grad.data

            if "vi" in self.attack_method:
                global_grad = torch.zeros_like(img_x)
                for i in range(self.sample_n):
                    r = torch.rand_like(img_x) * self.sample_beta * self.eps
                    r.requires_grad_()

                    outputs = self.model(img_x + delta + r)
                    
                    loss = self.loss_fn(outputs, y)
                    if self.target: 
                        loss = -loss
                    
                    loss.backward()
                    global_grad += r.grad.data
                    r.grad.data.zero_()

                current_grad = grad + variance
    
                # update variance
                variance = global_grad / self.sample_n - grad

                # return current_grad
                grad = current_grad

            # momentum
            if "mi" in self.attack_method or "ni" in self.attack_method:
                g = self.decay_factor * g + normalize_by_pnorm(grad, p=1)
                grad = g
            
            # Gaussian kernel
            if "ti" in self.attack_method:
                kernel = self.get_Gaussian_kernel(img_x)
                grad = F.conv2d(grad, kernel, padding=self.kernlen // 2)

            grad_sign = grad.data.sign()
            delta.data = delta.data + self.eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x.data + delta, 0., 1.) - x

            delta.grad.data.zero_()


        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        return x_adv




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

    def get_Gaussian_kernel(self, x):
        # define Gaussian kernel
        kern1d = st.norm.pdf(np.linspace(-self.nsig, self.nsig, self.kernlen))
        kernel = np.outer(kern1d, kern1d)
        kernel = kernel / kernel.sum()
        kernel = torch.FloatTensor(kernel).expand(x.size(1), x.size(1), self.kernlen, self.kernlen)
        kernel = kernel.to(x.device)
        return kernel

