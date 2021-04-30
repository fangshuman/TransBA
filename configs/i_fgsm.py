from .source_model import *

# i_fgsm
i_fgsm_base = {
    'eps': 16 / 255,
    'nb_iter': 10,
    'eps_iter': 1.6 / 255,
    'gamma': 1.,  # using sgm when gamma < 1.0
    'batch_size_coeff': 1.,
}


# ti-fgsm
ti_fgsm_base = {
    'kernlen': 7,
    'nsig': 3,
    **i_fgsm_base,
}


# di2-fgsm
di_fgsm_base = {
    'prob': 0.5,
    **i_fgsm_base,
}


# mi-fgsm
mi_fgsm_base = {
    'decay_factor': 1.0,
    **i_fgsm_base, 
}




fgsm_base = {
    'eps': 16 / 255,
    'nb_iter': 10,
    'eps_iter': 1.6 / 255,
    'gamma': 1.,  # using sgm when gamma < 1.0
    # 'batch_size_coeff': 1.,

    'prob': 0.5,

    'kernlen': 7,
    'nsig': 3,

    'decay_factor': 1.0,

    'scale_copies': 5,

    'vi_sample_n': 20,
    'vi_sample_beta': 1.5,

    'emi_sample_n': 11, 
    'emi_sample_eta': 7.,

}