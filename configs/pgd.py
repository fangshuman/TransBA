from .source_model import *

# pgd
pgd_base = {
    'eps': 16 / 255,
    'nb_iter': 10,
    'eps_iter': 1.6 / 255,
    'gamma': 1.,  # using sgm when gamma < 1.0
    'batch_size_coeff': 1.,
}

pgd_config = {
    'vgg16_pgd_config': {
        **pgd_base,
        **vgg16_config,
    },
    'resnet50_pgd_config': {
        **pgd_base,
        **resnet50_config,
    },
    'densenet121_pgd_config': {
        **pgd_base,
        **densenet121_config,
    },
    'inceptionv3_pgd_config': {
        **pgd_base,
        **inceptionv3_config,
    },
    'inceptionv4_pgd_config': {
        **pgd_base,
        **inceptionv4_config,
    },
    'inceptionresnetv2_pgd_config': {
        **pgd_base,
        **inceptionresnetv2_config,
    },
}



# ti-fgsm
ti_config_base = {
    'kernlen': 7,
    'nsig': 3,
    **pgd_base,
}

ti_config = {
    'vgg16_ti_config': {
        **ti_config_base,
        **vgg16_config,
    },
    'resnet50_ti_config': {
        **ti_config_base,
        **resnet50_config,
    },
    'densenet121_ti_config': {
        **ti_config_base,
        **densenet121_config,
    },
    'inceptionv3_ti_config': {
        **ti_config_base,
        **inceptionv3_config,
    },
    'inceptionv4_ti_config': {
        **ti_config_base,
        **inceptionv4_config,
    },
    'inceptionresnetv2_ti_config': {
        **ti_config_base,
        **inceptionresnetv2_config,
    },
}



# di2-fgsm
di_config_base = {
    'prob': 0.5,
    **pgd_base,
}

di_config = {
    'vgg16_di_config': {
        **di_config_base,
        **vgg16_config,
    },
    'resnet50_di_config': {
        **di_config_base,
        **resnet50_config,
    },
    'densenet121_di_config': {
        **di_config_base,
        'source_model_name': 'densenet121',
        'batch_size': 32,
    },
    'inceptionv3_di_config': {
        **di_config_base,
        **densenet121_config,
    },
    'inceptionv4_di_config': {
        **di_config_base,
        **inceptionv4_config,
    },
    'inceptionresnetv2_di_config': {
        **di_config_base,
        **inceptionresnetv2_config,
    },
}



# mi-fgsm
mi_config_base = {
    'decay_factor': 1.0,
    **pgd_base, 
}

mi_config = {
    'vgg16_mi_config': {
        **mi_config_base,
        **vgg16_config,
    },
    'resnet50_mi_config': {
        **mi_config_base,
        **resnet50_config,
    },
    'densenet121_mi_config': {
        **mi_config_base,
        'source_model_name': 'densenet121',
        'batch_size': 32,
    },
    'inceptionv3_mi_config': {
        **mi_config_base,
        **densenet121_config,
    },
    'inceptionv4_mi_config': {
        **mi_config_base,
        **inceptionv4_config,
    },
    'inceptionresnetv2_mi_config': {
        **mi_config_base,
        **inceptionresnetv2_config,
    },
}

