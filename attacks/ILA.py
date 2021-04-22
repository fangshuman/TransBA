import torch

from .base import I_FGSM_Attack


class ILAProjLoss(torch.nn.Module):
    def __init__(self):
        super(ILAProjLoss, self).__init__()
    def forward(self, old_attack_mid, new_mid, original_mid):
        n = old_attack_mid.shape[0]
        x = (old_attack_mid - original_mid).view(n, -1)  # y'-y
        y = (new_mid - original_mid).view(n, -1)         # y"-y
        # x_norm = x / torch.norm(x, dim = 1, keepdim = True)
        proj_loss = torch.sum(y * x) / n
        return proj_loss


class ILA_Attack(object):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, target=False):
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.target = target

    def ILA_forward(self, x):
        return x

    def perturb(self, x, y):
        # pgd attack
        pgd_nb_iter = self.nb_iter // 2
        delta = torch.zeros_like(x)
        delta.requires_grad_()
    
        for i in range(pgd_nb_iter):
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
    
        x_adv = torch.clamp(x + delta, 0., 1.)

        #import ipdb; ipdb.set_trace()

        # ila attack
        ila_nb_iter = self.nb_iter - pgd_nb_iter
        x_2 = x.clone()      # x"
        x_1 = x_adv.clone()  # x'
        x_0 = x.clone()      # x
        
        delta_ila = torch.zeros_like(x)
        delta_ila.requires_grad_()
        
        with torch.no_grad():
            mid_output_0 = self.ILA_forward(x_0)  # F_l(x)
            mid_output_1 = self.ILA_forward(x_1)  # F_l(x')
        for i in range(ila_nb_iter):
            mid_output_2 = self.ILA_forward(x_0 + delta_ila)
            loss = ILAProjLoss()(
                mid_output_1.detach(), 
                mid_output_2, 
                mid_output_0.detach(), 
            )
            if self.target:
                loss = -loss
            
            self.model.zero_grad()
            loss.backward()
            grad_sign = delta_ila.grad.data.sign()
            self.model.zero_grad()

            delta_ila.data = delta_ila.data + self.eps_iter * grad_sign
            delta_ila.data = torch.clamp(delta_ila.data, -self.eps, self.eps)
            delta_ila.data = torch.clamp(x_2.data + delta_ila, 0., 1.) - x_2

            delta_ila.grad.data.zero_()
        
        x_adv_2 = torch.clamp(x_2 + delta_ila, 0., 1.)

        return x_adv, x_2, x_adv_2



class ILA_Attack_for_ResNet(ILA_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, ila_layer='2_3', target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.ila_layer = ila_layer

    def ILA_forward(self, x):
        # conv1
        # bn1
        # relu          0_0
        # maxpool
        # layer1        1_*
        # layer2        2_*
        # layer3        3_*
        # layer4        4_*
        jj = int(self.ila_layer.split('_')[0])
        kk = int(self.ila_layer.split('_')[1])
        
        mean = torch.tensor(self.model.mean).view(3, 1, 1)
        std = torch.tensor(self.model.std).view(3, 1, 1)
        x = (x - mean.to(x.device)) / std.to(x.device)
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        if jj == 0 and kk ==0:
            return x
        x = self.model.maxpool(x)

        for ind, mm in enumerate(self.model.layer1):
            x = mm(x)
            if jj == 1 and ind == kk:
                return x
        for ind, mm in enumerate(self.model.layer2):
            x = mm(x)
            if jj == 2 and ind == kk:
                return x
        for ind, mm in enumerate(self.model.layer3):
            x = mm(x)
            if jj == 3 and ind == kk:
                return x
        for ind, mm in enumerate(self.model.layer4):
            x = mm(x)
            if jj == 4 and ind == kk:
                return x
        return False

    def perturb(self, x, y):
        return super().perturb(x, y)



class ILA_Attack_for_VGG(ILA_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, ila_layer='23', target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.ila_layer = int(ila_layer)

    def ILA_forward(self, x):
        # _features.0
        # _features.1
        # ...
        # _features.n
        mean = torch.tensor(self.model.mean).view(3, 1, 1)
        std = torch.tensor(self.model.std).view(3, 1, 1)
        x = (x - mean.to(x.device)) / std.to(x.device)

        for ind, mm in enumerate(self.model._features):
            x = mm(x)
            if ind == self.ila_layer:
                return x

    def perturb(self, x, y):
        return super().perturb(x, y)



class ILA_Attack_for_DenseNet(ILA_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, ila_layer='6', target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.ila_layer = int(ila_layer)
    
    def ILA_forward(self, x):
        # features.conv0            0
        # features.norm0            1
        # features.relu0         0  2
        # features.pool0            3
        # features.denseblock1   1  4
        # features.transition1   2  5
        # features.denseblock2   3  6
        # features.transition2   4  7
        # features.denseblock3   5  8
        # features.transition3   6  9
        # features.denseblock4   7  10
        # features.norm5         8  11
        mean = torch.tensor(self.model.mean).view(3, 1, 1)
        std = torch.tensor(self.model.std).view(3, 1, 1)
        x = (x - mean.to(x.device)) / std.to(x.device)

        for ind, mm in enumerate(self.model.features.module):
            x = mm(x)
            if self.ila_layer == 0 and ind == 2:
                return x
            elif self.ila_layer == ind + 3:
                return x
        return False

    def perturb(self, x, y):
        return super().perturb(x, y)



class ILA_Attack_for_InceptionV3(ILA_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, ila_layer='1_1', target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.ila_layer = ila_layer

    def ILA_forward(self, x):
        # Conv2d_1a     1_1
        # Conv2d_2a     2_1
        # Conv2d_2b     2_2
        # maxpool1
        # Conv2d_3b     3_2
        # Conv2d_4a     4_1
        # maxpool2
        # Mixed_5b      5_2
        # Mixed_5c      5_3
        # Mixed_5d      5_4
        # Mixed_6a      6_1
        # Mixed_6b      6_2
        # Mixed_6c      6_3
        # Mixed_6d      6_4
        # Mixed_6e      6_5
        # AuxLogits     
        # Mixed_7a      7_1
        # Mixed_7b      7_2
        # Mixed_7c      7_3
        # ...
        mean = torch.tensor(self.model.mean).view(3, 1, 1)
        std = torch.tensor(self.model.std).view(3, 1, 1)
        x = (x - mean.to(x.device)) / std.to(x.device)
        
        x = self.model.Conv2d_1a_3x3(x)
        if self.ila_layer == '1_1':
            return x
        
        x = self.model.Conv2d_2a_3x3(x)
        if self.ila_layer == '2_1':
            return x

        x = self.model.Conv2d_2b_3x3(x)
        if self.ila_layer == '2_2':
            return x

        x = self.maxpool1(x)

        x = self.model.Mixed_5b(x)
        if self.ila_layer == '5_2':
            return x

        x = self.model.Mixed_5c(x)
        if self.ila_layer == '5_3':
            return x

        x = self.model.Mixed_5d(x)
        if self.ila_layer == '5_4':
            return x

        x = self.model.Mixed_6a(x)
        if self.ila_layer == '6_1':
            return x

        x = self.model.Mixed_6b(x)
        if self.ila_layer == '6_2':
            return x
        
        x = self.model.Mixed_6c(x)
        if self.ila_layer == '6_3':
            return x

        x = self.model.Mixed_6d(x)
        if self.ila_layer == '6_4':
            return x

        x = self.model.Mixed_6e(x)
        if self.ila_layer == '6_5':
            return x

        x = self.AuxLogits(x)

        x = self.model.Mixed_7a(x)
        if self.ila_layer == '7_1':
            return x
        
        x = self.model.Mixed_7b(x)
        if self.ila_layer == '7_2':
            return x

        x = self.model.Mixed_7c(x)
        if self.ila_layer == '7_3':
            return x

    def perturb(self, x):
        return super().perturb(x, y)



class ILA_Attack_for_InceptionV4(ILA_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, ila_layer='0', target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.ila_layer = int(ila_layer)

    def ILA_forward(self, x):
        # features.BasicConv2d      0
        # features.BasicConv2d      1
        # features.BasicConv2d      2
        # features.Mixed_3a()       3
        # features.Mixed_4a()       4
        # features.Mixed_5a()       5
        # features.Inception_A()    6
        # features.Inception_A()    7
        # features.Inception_A()    8
        # features.Inception_A()    9
        # features.Reduction_A()    10
        # features.Inception_B()    11
        # features.Inception_B()    12
        # features.Inception_B()    13
        # features.Inception_B()    14
        # features.Inception_B()    15
        # features.Inception_B()    16
        # features.Inception_B()    17
        # features.Reduction_B()    18
        # features.Inception_C()    19
        # features.Inception_C()    20
        # features.Inception_C()    21
        mean = torch.tensor(self.model.mean).view(3, 1, 1)
        std = torch.tensor(self.model.std).view(3, 1, 1)
        x = (x - mean.to(x.device)) / std.to(x.device)

        for ind, mm in enumerate(self.model.features.module):
            x = mm(x)
            if ind == self.ila_layer:
                return x

    def perturb(self, x):
        return super().perturb(x, y)



class ILA_Attack_for_InceptionResNetV2(ILA_Attack):
    def __init__(self, model, loss_fn, eps=0.05, nb_iter=10, eps_iter=0.005, ila_layer='23', target=False):
        super().__init__(model, loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, target=target)
        self.ila_layer = ila_layer

    def ILA_forward(self, x):
        # conv2d_1a     1_1
        # conv2d_2a     2_1
        # conv2d_2b     2_2
        # maxpool_3a    
        # conv2d_3b     3_2
        # conv2d_4a     4_1
        # maxpool_5a    
        # mixed_5b      5_2
        # repeat        
        # mixed_6a      6_1
        # repeat_1      
        # mixed_7a      7_1
        # repeat_2      
        # block8        
        # conv2d_7b     7_2
        mean = torch.tensor(self.model.mean).view(3, 1, 1)
        std = torch.tensor(self.model.std).view(3, 1, 1)
        x = (x - mean.to(x.device)) / std.to(x.device)

        x = self.conv2d_1a(x)
        if self.ila_layer == '1_1':
            return x
        
        x = self.conv2d_2a(x)
        if self.ila_layer == '2_1':
            return x

        x = self.conv2d_2b(x)
        if self.ila_layer == '2_2':
            return x

        x = self.maxpool_3a(x)

        x = self.conv2d_3b(x)
        if self.ila_layer == '3_2':
            return x

        x = self.conv2d_4a(x)
        if self.ila_layer == '4_1':
            return x

        x = self.maxpool_5a(x)

        x = self.mixed_5b(x)
        if self.ila_layer == '5_2':
            return x

        x = self.repeat(x)

        x = self.mixed_6a(x)
        if self.ila_layer == '6_1':
            return x

        x = self.repeat_1(x)

        x = self.mixed_7a(x)
        if self.ila_layer == '7_1':
            return x

        x = self.repeat_2(x)
        x = self.blaock8(x)

        x = self.mixed_7b(x)
        if self.ila_layer == '7_2':
            return x

    def perturb(self, x, y):
        return super().perturb(x, y)

