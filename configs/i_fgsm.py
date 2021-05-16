fgsm_base = {
    'eps': 16 / 255,
    'nb_iter': 10,
    'eps_iter': 1.6 / 255,
    'gamma': 1.,  # using sgm when gamma < 1.0

    'prob': 0.5,            # for di

    'kernlen': 7,           # for ti
    'nsig': 3,              # for ti

    'decay_factor': 1.0,    # for mi/ni

    'scale_copies': 5,      # for si

    'vi_sample_n': 20,      # for vi
    'vi_sample_beta': 1.5,  # for vi

    # 'emi_sample_n': 11,     # for emi
    # 'emi_sample_eta': 7.,   # for emi

    'amplification': 10,    # for pi (patch-wise)

}