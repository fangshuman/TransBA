# attack method config

attack_base_config = {
    'eps': 16 / 255,
    'nb_iter': 10,
    'eps_iter': 1.6 / 255,
}

i_fgsm_config = {
    **attack_base_config,
    'attack_method': 'i-fgsm',
}

ti_fgsm_config = {
    **attack_base_config,
    'attack_method': 'ti-fgsm',
    'kernlen': 7,
    'nsig': 3,
}

di_fgsm_config = {
    **attack_base_config,
    'attack_method': 'di-fgsm',
    'prob': 0.5,
}

mi_fgsm_config = {
    **attack_base_config,
    'attack_method': 'mi-fgsm',
    'decay_factor': 1.0,
}