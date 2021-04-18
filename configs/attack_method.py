# attack method 
attack_methods = [
    'i_fgsm',
    'ti_fgsm',
    'di_fgsm',
    'mi_fgsm',
    'TAP',
    'Ghost',
    'SGM',
    'LinBP',
]


# attack method config
attack_base_config = {
    'eps': 16 / 255,
    'nb_iter': 10,
    'eps_iter': 1.6 / 255,
}

i_fgsm_config = {
    **attack_base_config,
    'attack_method': 'i_fgsm',
}

ti_fgsm_config = {
    **attack_base_config,
    'attack_method': 'ti_fgsm',
    'kernlen': 7,
    'nsig': 3,
}

di_fgsm_config = {
    **attack_base_config,
    'attack_method': 'di_fgsm',
    'prob': 0.5,
}

mi_fgsm_config = {
    **attack_base_config,
    'attack_method': 'mi_fgsm',
    'decay_factor': 1.0,
}