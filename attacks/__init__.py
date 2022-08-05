from .base_attacker import Based_Attacker
from .ila import ILA_Attacker
from .patch_wise import Patchwise_Attacker
from .sgm import SGM_Attacker
from .admix import Admix_Attacker
from .emi import EMI_Attacker
from .vt import VT_Attacker
from .fia import FIA_Attacker

attack_map = {
    "fgsm": Based_Attacker,
    "ila": ILA_Attacker,
    "patchwise": Patchwise_Attacker,
    "sgm": SGM_Attacker,
    "admix": Admix_Attacker,
    "emi": EMI_Attacker,
    "vt": VT_Attacker,
    "fia": FIA_Attacker,
}


def get_attack(attack_method, model=None, loss_fn=None, args=None):
    def _get_attack(attack_method):
        if attack_method in attack_map:
            return attack_map[attack_method]
        elif attack_method.startswith("patchwise"):
            return Patchwise_Attacker
        elif attack_method.startswith("sgm"):
            return SGM_Attacker
        elif attack_method.startswith("admix"):
            return Admix_Attacker
        elif attack_method.startswith("emi"):
            return EMI_Attacker
        elif attack_method.startswith("vt"):
            return VT_Attacker
        elif attack_method.startswith("fia"):
            return FIA_Attacker
        elif attack_method.endswith("fgsm"):
            return Based_Attacker
        else:
            raise NotImplementedError(f"No such attack method: {attack_method}")

    attack = _get_attack(attack_method)
    if model is None:
        return attack
    return attack(attack_method, model, loss_fn, args)
