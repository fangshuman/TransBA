from .source_model import *

sgm_base = {
    'gamma': 0.5,
    'eps': 16 / 255,
    'nb_iter': 10,
    'eps_iter': 2.0 / 255,
    'batch_size_coeff': 1.,  # reduce batch size
}

sgm_config = {
    'vgg16_sgm_config': {
        **sgm_base,
        **vgg16_config,
    },
    'resnet50_sgm_config': {
         **sgm_base,
         **resnet50_config,
    },
    'densenet121_sgm_config': {
        **sgm_base,
        **densenet121_config,
    },
    'inceptionv3_sgm_config': {
        **sgm_base,
        **inceptionv3_config,
        
    },
    'inceptionv4_sgm_config': {
        **sgm_base, 
        **inceptionv4_config,
        
    },
    'inceptionresnetv2_sgm_config': {
        **sgm_base,
        **inceptionresnetv2_config,
    },
}