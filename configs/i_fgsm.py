from .source_model import *

# i_fgsm
i_fgsm_base = {
    'eps': 16 / 255,
    'nb_iter': 10,
    'eps_iter': 1.6 / 255,
    'gamma': 1.,  # using sgm when gamma < 1.0
    'batch_size_coeff': 1.,
}

# i_fgsm_config = {
#     'vgg16_i_fgsm_config': {
#         **i_fgsm_base,
#         **vgg16_config,
#     },
#     'resnet50_i_fgsm_config': {
#         **i_fgsm_base,
#         **resnet50_config,
#     },
#     'densenet121_i_fgsm_config': {
#         **i_fgsm_base,
#         **densenet121_config,
#     },
#     'inceptionv3_i_fgsm_config': {
#         **i_fgsm_base,
#         **inceptionv3_config,
#     },
#     'inceptionv4_i_fgsm_config': {
#         **i_fgsm_base,
#         **inceptionv4_config,
#     },
#     'inceptionresnetv2_i_fgsm_config': {
#         **i_fgsm_base,
#         **inceptionresnetv2_config,
#     },
# }

# i_fgsm_awp_config = {
#     'vgg16_i_fgsm_awp_config': {
#         **i_fgsm_base,
#         **vgg16_config,
#         'awp': True,
#     },
#     'resnet50_i_fgsm_awp_config': {
#         **i_fgsm_base,
#         **resnet50_config,
#         'awp': True,
#     },
#     'densenet121_i_fgsm_awp_config': {
#         **i_fgsm_base,
#         **densenet121_config,
#         'awp': True,
#     },
#     'inceptionv3_i_fgsm_awp_config': {
#         **i_fgsm_base,
#         **inceptionv3_config,
#         'awp': True,
#     },
#     'inceptionv4_i_fgsm_awp_config': {
#         **i_fgsm_base,
#         **inceptionv4_config,
#         'awp': True,
#     },
#     'inceptionresnetv2_i_fgsm_awp_config': {
#         **i_fgsm_base,
#         **inceptionresnetv2_config,
#         'awp': True,
#     },
# }



# ti-fgsm
ti_fgsm_base = {
    'kernlen': 7,
    'nsig': 3,
    **i_fgsm_base,
}

# ti_fgsm_config = {
#     'vgg16_ti_fgsm_config': {
#         **ti_fgsm_base,
#         **vgg16_config,
#     },
#     'resnet50_ti_fgsm_config': {
#         **ti_fgsm_base,
#         **resnet50_config,
#     },
#     'densenet121_ti_fgsm_config': {
#         **ti_fgsm_base,
#         **densenet121_config,
#     },
#     'inceptionv3_ti_fgsm_config': {
#         **ti_fgsm_base,
#         **inceptionv3_config,
#     },
#     'inceptionv4_ti_fgsm_config': {
#         **ti_fgsm_base,
#         **inceptionv4_config,
#     },
#     'inceptionresnetv2_ti_fgsm_config': {
#         **ti_fgsm_base,
#         **inceptionresnetv2_config,
#     },
# }

# ti_fgsm_awp_config = {
#     'vgg16_ti_fgsm_awp_config': {
#         **ti_fgsm_base,
#         **vgg16_config,
#         'awp': True,
#     },
#     'resnet50_ti_fgsm_awp_config': {
#         **ti_fgsm_base,
#         **resnet50_config,
#         'awp': True,
#     },
#     'densenet121_ti_fgsm_awp_config': {
#         **ti_fgsm_base,
#         **densenet121_config,
#         'awp': True,
#     },
#     'inceptionv3_ti_fgsm_awp_config': {
#         **ti_fgsm_base,
#         **inceptionv3_config,
#         'awp': True,
#     },
#     'inceptionv4_ti_fgsm_awp_config': {
#         **ti_fgsm_base,
#         **inceptionv4_config,
#         'awp': True,
#     },
#     'inceptionresnetv2_ti_fgsm_awp_config': {
#         **ti_fgsm_base,
#         **inceptionresnetv2_config,
#         'awp': True,
#     },
# }




# di2-fgsm
di_fgsm_base = {
    'prob': 0.5,
    **i_fgsm_base,
}

# di_fgsm_config = {
#     'vgg16_di_fgsm_config': {
#         **di_fgsm_base,
#         **vgg16_config,
#     },
#     'resnet50_di_fgsm_config': {
#         **di_fgsm_base,
#         **resnet50_config,
#     },
#     'densenet121_di_fgsm_config': {
#         **di_fgsm_base,
#         **densenet121_config,
#     },
#     'inceptionv3_di_fgsm_config': {
#         **di_fgsm_base,
#         **inceptionv3_config,
#     },
#     'inceptionv4_di_fgsm_config': {
#         **di_fgsm_base,
#         **inceptionv4_config,
#     },
#     'inceptionresnetv2_di_fgsm_config': {
#         **di_fgsm_base,
#         **inceptionresnetv2_config,
#     },
# }

# di_fgsm_awp_config = {
#     'vgg16_di_fgsm_awp_config': {
#         **di_fgsm_base,
#         **vgg16_config,
#         'awp': True,
#     },
#     'resnet50_di_fgsm_awp_config': {
#         **di_fgsm_base,
#         **resnet50_config,
#         'awp': True,
#     },
#     'densenet121_di_fgsm_awp_config': {
#         **di_fgsm_base,
#         **densenet121_config,
#         'awp': True,
#     },
#     'inceptionv3_di_fgsm_awp_config': {
#         **di_fgsm_base,
#         **inceptionv3_config,
#         'awp': True,
#     },
#     'inceptionv4_di_fgsm_awp_config': {
#         **di_fgsm_base,
#         **inceptionv4_config,
#         'awp': True,
#     },
#     'inceptionresnetv2_di_fgsm_awp_config': {
#         **di_fgsm_base,
#         **inceptionresnetv2_config,
#         'awp': True,
#     },
# }



# mi-fgsm
mi_fgsm_base = {
    'decay_factor': 1.0,
    **i_fgsm_base, 
}

# mi_fgsm_config = {
#     'vgg16_mi_fgsm_config': {
#         **mi_fgsm_base,
#         **vgg16_config,
#     },
#     'resnet50_mi_fgsm_config': {
#         **mi_fgsm_base,
#         **resnet50_config,
#     },
#     'densenet121_mi_fgsm_config': {
#         **mi_fgsm_base,
#         **densenet121_config,
#     },
#     'inceptionv3_mi_fgsm_config': {
#         **mi_fgsm_base,
#         **inceptionv3_config,
#     },
#     'inceptionv4_mi_fgsm_config': {
#         **mi_fgsm_base,
#         **inceptionv4_config,
#     },
#     'inceptionresnetv2_mi_fgsm_config': {
#         **mi_fgsm_base,
#         **inceptionresnetv2_config,
#     },
# }

# mi_fgsm_awp_config = {
#     'vgg16_mi_fgsm_awp_config': {
#         **mi_fgsm_base,
#         **vgg16_config,
#         'awp': True,
#     },
#     'resnet50_mi_fgsm_awp_config': {
#         **mi_fgsm_base,
#         **resnet50_config,
#         'awp': True,
#     },
#     'densenet121_mi_fgsm_awp_config': {
#         **mi_fgsm_base,
#         **densenet121_config,
#         'awp': True,
#     },
#     'inceptionv3_mi_fgsm_awp_config': {
#         **mi_fgsm_base,
#         **inceptionv3_config,
#         'awp': True,
#     },
#     'inceptionv4_mi_fgsm_awp_config': {
#         **mi_fgsm_base,
#         **inceptionv4_config,
#         'awp': True,
#     },
#     'inceptionresnetv2_mi_fgsm_awp_config': {
#         **mi_fgsm_base,
#         **inceptionresnetv2_config,
#         'awp': True,
#     },
# }
