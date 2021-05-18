from .basic import *
from .sgm import *
from .ila import *
from .ni_si_fgsm import *
from .vi_fgsm import *
from .multi import *


def get_attack(attack, arch, model, loss_fn, args):
    if attack.endswith('fgsm'):
        return Multi_Attack(attack, model, loss_fn, args)

    elif 'sgm' in attack:
        return SGM_Attack(
            arch=arch,
            model=model,
            loss_fn=loss_fn,
            args=args,
        )

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

    else:
        raise NotImplementedError(f"No such attack method: {attack}")
