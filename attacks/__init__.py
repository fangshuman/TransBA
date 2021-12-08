from .attacker import IFGSM_Based_Attacker
from .sgm import SGM_Attack
from .ila import ILA_Attack
from .topo import Topo

attack_map = {
    "fgsm": IFGSM_Based_Attacker,
    "sgm": SGM_Attack,
    "ila": ILA_Attack,
    "topo": Topo,
}


def get_attack(attack_method, model=None, loss_fn=None, args=None):
    def _get_attack(attack_method):
        if attack_method in attack_map:
            return attack_map[attack_method]
        elif attack_method.endswith("fgsm"):
            return IFGSM_Based_Attacker
        else:
            raise NotImplementedError(f"No such attack method: {attack_method}")

    attack = _get_attack(attack_method)
    if model is None:
        return attack
    return attack(attack_method, model, loss_fn, args)
