from .attacker import IFGSM_Based_Attacker
from .admix import Admix_Attacker
from .sgm import SGM_Attacker
from .ila import ILA_Attacker
from .fia import FIA_Attacker
from .topo import Topo

attack_map = {
    "fgsm": IFGSM_Based_Attacker,
    "sgm": SGM_Attacker,
    "ila": ILA_Attacker,
    "fia": FIA_Attacker,
    "topo": Topo,
}


def get_attack(attack_method, model=None, loss_fn=None, args=None):
    def _get_attack(attack_method):
        if attack_method in attack_map:
            return attack_map[attack_method]
        elif attack_method.startswith("sgm"):
            return SGM_Attacker
        elif attack_method.startswith("admix"):
            return Admix_Attacker
        elif attack_method.startswith("fia"):
            return FIA_Attacker
        elif attack_method.endswith("fgsm"):
            return IFGSM_Based_Attacker
        else:
            raise NotImplementedError(f"No such attack method: {attack_method}")

    attack = _get_attack(attack_method)
    if model is None:
        return attack
    return attack(attack_method, model, loss_fn, args)
