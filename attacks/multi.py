from .basic import *
from .ni_si_fgsm import *
from .vi_fgsm import *
from .utils import normalize_by_pnorm

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

        # try:
        #     # basic
        #     self.eps = args.eps
        #     self.nb_iter = args.nb_iter
        #     self.eps_iter = args.eps_iter
        #     self.target = args.target
        #     # extra
        #     self.prob = args.prob
        #     self.kernlen = args.kernlen
        #     self.nsig = args.nsig
        #     self.decay_factor = args.decay_factor
        #     self.scale_copies = args.scale_copies
        #     self.sample_n = args.vi_sample_n
        #     self.sample_beta = args.vi_sample_beta
        #     self.amplification = args.amplification

        # except:
        #     # basic default value
        #     self.eps = 0.05
        #     self.nb_iter = 10
        #     self.eps_iter = 0.005
        #     self.target = False
        #     # extra default value
        #     self.prob = 0.5
        #     self.kernlen = 7
        #     self.nsig = 3
        #     self.decay_factor = 1.0
        #     self.scale_copies = 5
        #     self.sample_n = 20
        #     self.sample_beta = 1.5
        #     self.amplification = 10

        default_value = {
            # basic default value
            'eps': 0.05,
            'nb_iter': 10, 
            'eps_iter': 0.005,
            'target': False,
            # extra default value
            'prob': 0.5,
            'kernlen': 7,
            'nsig': 3,
            'decay_factor': 1.0,
            'scale_copies': 5,
            'sample_n': 20,
            'sample_beta': 1.5,
            'amplification': 10,
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

        # initialize extra var
        if "mi" in self.attack_method or "ni" in self.attack_method:
            g = torch.zeros_like(x)
        if "vi" in self.attack_method:
            variance = torch.zeros_like(x)
        if "pi" in self.attack_method:
            a = torch.zeros_like(x)
            eps_iter *= self.amplification
            stack_kern, kern_size = self.project_kern(3)

        extra_item = torch.zeros_like(x)
        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.nb_iter):
            if "ni" in self.attack_method: 
                img_x = x + self.decay_factor * eps_iter * g 
            else:
                img_x = x
            
            # get gradient
            if "si" in self.attack_method:
                grad = torch.zeros_like(img_x)
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
                # get average value of gradient
                grad = grad / self.scale_copies

            else:
                if "di" in self.attack_method:
                    outputs = self.model(self.input_diversity(img_x + delta))
                else:
                    outputs = self.model(img_x + delta)
                
                loss = self.loss_fn(outputs, y)
                if self.target:
                    loss = -loss
            
                loss.backward()
                grad = delta.grad.data

            # variance: VI-FGSM
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


            # Gaussian kernel: TI-FGSM
            if "ti" in self.attack_method:
                kernel = self.get_Gaussian_kernel(img_x)
                grad = F.conv2d(grad, kernel, padding=self.kernlen // 2)


            # momentum: MI-FGSM / NI-FGSM
            if "mi" in self.attack_method or "ni" in self.attack_method:
                g = self.decay_factor * g + normalize_by_pnorm(grad, p=1)
                grad = g

            
            # Patch-wise attach: PI-FGSM
            if "pi" in self.attack_method:
                a += eps_iter * grad.data.sign()
                cut_noise = torch.clamp(abs(a) - self.eps, 0, 1e5) * a.sign()
                projection = eps_iter * (self.project_noise(cut_noise, stack_kern, kern_size)).sign()
                a += projection
                extra_item = projection  # return extra item


            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign + extra_item
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

    def project_kern(self, kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).cuda()
        return stack_kern, kern_size // 2

    def project_noise(self, x, stack_kern, kern_size):
        x = F.conv2d(x, stack_kern, padding = (kern_size, kern_size), groups=3)
        return x



