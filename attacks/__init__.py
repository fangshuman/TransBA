from .base import *
from .SGM import *
from .ILA import *
from .basic import *

from advertorch.attacks import LinfPGDAttack

def get_attack(attack, model, loss_fn, args):
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
                              target=args.target,)

    elif attack == 'di_fgsm':
        return DI_FGSM_Attack(model=model,
                              loss_fn=loss_fn,
                              eps=args.eps,
                              nb_iter=args.nb_iter,
                              eps_iter=args.eps_iter,
                              prob=args.prob,
                              target=args.target,)

    elif attack == 'mi_fgsm':
        return MI_FGSM_Attack(model=model,
                              loss_fn=loss_fn,
                              eps=args.eps,
                              nb_iter=args.nb_iter,
                              eps_iter=args.eps_iter,
                              decay_factor=args.decay_factor,
                              target=args.target,)

    elif attack == 'ila':
        return ILA_Attack(model_name=args.source_model,
                          model=model,
                          loss_fn=loss_fn,
                          ila_layer=args.ila_layer,
                          eps=args.eps,
                          nb_iter=args.nb_iter,
                          step_size_pgd=args.step_size_pgd,
                          step_size_ila=args.step_size_ila,
                          target=args.target)

    elif attack == 'sgm':
        if args.source_model == 'resnet50':
            return SGM_Attack_for_ResNet(model=model,
                                         loss_fn=loss_fn,
                                         eps=args.eps,
                                         nb_iter=args.nb_iter,
                                         eps_iter=args.eps_iter,
                                         gamma=args.gamma,
                                         target=args.target,)
        elif args.source_model == 'densenet121':
            return SGM_Attack_for_DenseNet(model=model,
                                           loss_fn=loss_fn,
                                           eps=args.eps,
                                           nb_iter=args.nb_iter,
                                           eps_iter=args.eps_iter,
                                           gamma=args.gamma,
                                           target=args.target,)
        else:
            raise NotImplementedError("Current code only supports resnet50/densenet121. Please check souce model name.")

    elif attack == 'linbp':
        pass

    else:
        raise NotImplementedError(f"No such attack method: {attack}")
