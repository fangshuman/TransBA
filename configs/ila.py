from .source_model import *

ila_base = {
    'eps': 16 / 255,
    'nb_iter': 20,
    'step_size_pgd': 0.008,
    'step_size_ila': 0.01,
    'gamma': 1.,   # using sgm when gamma < 1.0
    'batch_size_coeff': 0.5,  # reduce batch size
}

ila_config = {
    'vgg16_ila_config': {
        'ila_layer': 28,  # choose from 0...43
        **ila_base,
        **vgg16_config,
    },
    'resnet50_ila_config': {
        'ila_layer': 4,  # choose from 0...7
         **ila_base,
         **resnet50_config,
    },
    'densenet121_ila_config': {
        'ila_layer': 6,   # choose from 0...9
        **ila_base,
        **densenet121_config,
    },
    'inceptionv3_ila_config': {
        'ila_layer': 4,     # choose from 0...10
        **ila_base,
        **inceptionv3_config,
        
    },
    'inceptionv4_ila_config': {
        'ila_layer': 6,   # chose from 0...21
        **ila_base, 
        **inceptionv4_config,
        
    },
    'inceptionresnetv2_ila_config': {
        'ila_layer': 5,     # choose from 0...17
        **ila_base,
        **inceptionresnetv2_config,
    },
}