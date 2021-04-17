from .base import *


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
                              kerlen=args.kernlen,
                              nsig=args.nsig,
                              target=args.target)
    elif attack == 'mi_fgsm':
        return MI_FGSM_Attack(model=model, 
                              loss_fn=loss_fn,
                              eps=args.eps,
                              nb_iter=args.nb_iter,
                              eps_iter=args.eps_iter,
                              decay_factor=args.decay_factor,
                              target=args.target)
    else:
        raise NotImplementedError(f"No such attack method: {attack}")