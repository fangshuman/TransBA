from .source_model import *

sgm_base = {
    'eps': 16 / 255,
    'nb_iter': 10,
    'eps_iter': 1.6 / 255,
    'batch_size_coeff': 1.,  # reduce batch size
}

sgm_config = {
    'resnet50_sgm_config': {
        'gamma': 0.2, 
        **sgm_base,
        **resnet50_config,
    },
    'densenet121_sgm_config': {
        'gamma': 0.5,
        **sgm_base,
        **densenet121_config,
    },

}