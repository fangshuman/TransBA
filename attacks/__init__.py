from .base import *
from .SGM import *
from .ILA import *


def get_attack(attack, model, loss_fn, args):
    if attack == 'pgd':
        return I_FGSM_Attack(model=model, 
                             loss_fn=loss_fn,
                             eps=args.eps,
                             nb_iter=args.nb_iter,
                             eps_iter=args.eps_iter,
                             target=args.target)
    
    elif attack == 'ti':
        return TI_FGSM_Attack(model=model, 
                              loss_fn=loss_fn,
                              eps=args.eps,
                              nb_iter=args.nb_iter,
                              eps_iter=args.eps_iter,
                              kernlen=args.kernlen,
                              nsig=args.nsig,
                              target=args.target)
    
    elif attack == 'di':
        return DI_FGSM_Attack(model=model, 
                              loss_fn=loss_fn,
                              eps=args.eps,
                              nb_iter=args.nb_iter,
                              eps_iter=args.eps_iter,
                              prob=args.prob,
                              target=args.target)
    
    elif attack == 'mi':
        return MI_FGSM_Attack(model=model, 
                              loss_fn=loss_fn,
                              eps=args.eps,
                              nb_iter=args.nb_iter,
                              eps_iter=args.eps_iter,
                              decay_factor=args.decay_factor,
                              target=args.target)

    elif attack == 'ila':
        if 'vgg' in args.source_model:
            return ILA_Attack_for_VGG(model=model, 
                                      loss_fn=loss_fn,
                                      eps=args.eps,
                                      nb_iter=args.nb_iter,
                                      eps_iter=args.eps_iter,
                                      ila_layer=args.ila_layer,
                                      target=args.target)
        elif 'resnet' in args.source_model:
            return ILA_Attack_for_ResNet(model=model, 
                                         loss_fn=loss_fn,
                                         eps=args.eps,
                                         nb_iter=args.nb_iter,
                                         eps_iter=args.eps_iter,
                                         ila_layer=args.ila_layer,
                                         target=args.target)
        elif 'densenet' in args.source_model:
            return ILA_Attack_for_DenseNet(model=model, 
                                           loss_fn=loss_fn,
                                           eps=args.eps,
                                           nb_iter=args.nb_iter,
                                           eps_iter=args.eps_iter,
                                           ila_layer=args.ila_layer,
                                           target=args.target)
        elif args.source_model == 'inceptionv3':
            return ILA_Attack_for_InceptionV3(model=model, 
                                              loss_fn=loss_fn,
                                              eps=args.eps,
                                              nb_iter=args.nb_iter,
                                              eps_iter=args.eps_iter,
                                              ila_layer=args.ila_layer,
                                              target=args.target)
        elif args.source_model == 'inceptionv4':
            return ILA_Attack_for_InceptionV4(model=model, 
                                              loss_fn=loss_fn,
                                              eps=args.eps,
                                              nb_iter=args.nb_iter,
                                              eps_iter=args.eps_iter,
                                              ila_layer=args.ila_layer,
                                              target=args.target)
        elif args.source_model == 'inceptionresnetv2':
            return ILA_Attack_for_InceptionResNetV2(model=model, 
                                                    loss_fn=loss_fn,
                                                    eps=args.eps,
                                                    nb_iter=args.nb_iter,
                                                    eps_iter=args.eps_iter,
                                                    ila_layer=args.ila_layer,
                                                    target=args.target)
        else:
            raise NotImplementedError("Current code only supports vgg/resnet/densenet/inceptionv3/inceptionv4/inceptionresnetv2. Please check souce model name.") 

    elif attack == 'sgm':
        if 'vgg' in args.source_model:
            return SGM_Attack_for_VGG(model=model, 
                                      loss_fn=loss_fn,
                                      eps=args.eps,
                                      nb_iter=args.nb_iter,
                                      eps_iter=args.eps_iter,
                                      gamma=args.gamma,
                                      target=args.target)
        elif 'resnet' in args.source_model:
            return SGM_Attack_for_ResNet(model=model, 
                                         loss_fn=loss_fn,
                                         eps=args.eps,
                                         nb_iter=args.nb_iter,
                                         eps_iter=args.eps_iter,
                                         gamma=args.gamma,
                                         target=args.target)
        elif 'densenet' in args.source_model:
            return SGM_Attack_for_DenseNet(model=model, 
                                           loss_fn=loss_fn,
                                           eps=args.eps,
                                           nb_iter=args.nb_iter,
                                           eps_iter=args.eps_iter,
                                           gamma=args.gamma,
                                           target=args.target)
        elif args.source_model == 'inceptionv3':
            return SGM_Attack_for_InceptionV3(model=model, 
                                              loss_fn=loss_fn,
                                              eps=args.eps,
                                              nb_iter=args.nb_iter,
                                              eps_iter=args.eps_iter,
                                              gamma=args.gamma,
                                              target=args.target)
        elif args.source_model == 'inceptionv4':
            return SGM_Attack_for_InceptionV4(model=model, 
                                              loss_fn=loss_fn,
                                              eps=args.eps,
                                              nb_iter=args.nb_iter,
                                              eps_iter=args.eps_iter,
                                              gamma=args.gamma,
                                              target=args.target)
        elif args.source_model == 'inceptionresnetv2':
            return SGM_Attack_for_InceptionResNetV2(model=model, 
                                                    loss_fn=loss_fn,
                                                    eps=args.eps,
                                                    nb_iter=args.nb_iter,
                                                    eps_iter=args.eps_iter,
                                                    gamma=args.gamma,
                                                    target=args.target)
        else:
            raise NotImplementedError("Current code only supports vgg/resnet/densenet/inceptionv3/inceptionv4/inceptionresnetv2. Please check souce model name.")
    
    elif attack == 'linbp':
        pass

    else:
        raise NotImplementedError(f"No such attack method: {attack}")