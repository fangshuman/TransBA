from .base import *
from .SGM import *


def get_attack(attack, model_name, model, loss_fn, args):
    if attack == 'i_fgsm':
        return I_FGSM_Attack(model=model, 
                             loss_fn=loss_fn,
                             eps=args.eps,
                             nb_iter=args.nb_iter,
                             eps_iter=args.eps_iter,
                             target=args.target)
    
    elif attack == 'ti_fgsm':
        return TI_FGSM_Attack(model=model, 
                              loss_fn=loss_fn,
                              eps=args.eps,
                              nb_iter=args.nb_iter,
                              eps_iter=args.eps_iter,
                              kernlen=args.kernlen,
                              nsig=args.nsig,
                              target=args.target)
    
    elif attack == 'di_fgsm':
        return DI_FGSM_Attack(model=model, 
                              loss_fn=loss_fn,
                              eps=args.eps,
                              nb_iter=args.nb_iter,
                              eps_iter=args.eps_iter,
                              prob=args.prob,
                              target=args.target)
    
    elif attack == 'mi_fgsm':
        return MI_FGSM_Attack(model=model, 
                              loss_fn=loss_fn,
                              eps=args.eps,
                              nb_iter=args.nb_iter,
                              eps_iter=args.eps_iter,
                              decay_factor=args.decay_factor,
                              target=args.target)

    elif attack == 'SGM':
        if 'vgg' in model_name:
            return SGM_Attack_for_VGG(model=model, 
                                      loss_fn=loss_fn,
                                      eps=args.eps,
                                      nb_iter=args.nb_iter,
                                      eps_iter=args.eps_iter,
                                      gamma=args.gamma,
                                      target=args.target)
        elif 'resnet' in model_name:
            return SGM_Attack_for_ResNet(model=model, 
                                         loss_fn=loss_fn,
                                         eps=args.eps,
                                         nb_iter=args.nb_iter,
                                         eps_iter=args.eps_iter,
                                         gamma=args.gamma,
                                         target=args.target)
        elif 'densenet' in model_name:
            return SGM_Attack_for_DenseNet(model=model, 
                                           loss_fn=loss_fn,
                                           eps=args.eps,
                                           nb_iter=args.nb_iter,
                                           eps_iter=args.eps_iter,
                                           gamma=args.gamma,
                                           target=args.target)
        elif model_name == 'inceptionv3':
            return SGM_Attack_for_InceptionV3(model=model, 
                                              loss_fn=loss_fn,
                                              eps=args.eps,
                                              nb_iter=args.nb_iter,
                                              eps_iter=args.eps_iter,
                                              gamma=args.gamma,
                                              target=args.target)
        elif model_name == 'inceptionv4':
            return SGM_Attack_for_InceptionV4(model=model, 
                                              loss_fn=loss_fn,
                                              eps=args.eps,
                                              nb_iter=args.nb_iter,
                                              eps_iter=args.eps_iter,
                                              gamma=args.gamma,
                                              target=args.target)
        elif model_name == 'inceptionresnetv2':
            return SGM_Attack_for_InceptionResNetV2(model=model, 
                                                    loss_fn=loss_fn,
                                                    eps=args.eps,
                                                    nb_iter=args.nb_iter,
                                                    eps_iter=args.eps_iter,
                                                    gamma=args.gamma,
                                                    target=args.target)
        else:
            raise NotImplementedError("Current code only supports vgg/resnet/densenet/inc_v3/inc_v4/inc_resv2. Please check souce model name.")
    
    elif attack == 'LinBP':
        pass

    else:
        raise NotImplementedError(f"No such attack method: {attack}")