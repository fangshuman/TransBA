from .source_model import *

ila_base = {
    'eps': 16,
    'nb_iter': 20,
    'eps_iter': 1.6,
    'gamma': 1.,  # using sgm when gamma < 1.0
}

ila_config = {
    'vgg16_ila_config': {
        **ila_base,
        **vgg16_config,
        'ila_layer': '23',  # choose from 0...43
    },
    'resnet50_ila_config': {
         **ila_base,
         **resnet50_config,
         'ila_layer': '2_3',  # choose from 0_0, 1_*, 2_*, 3_*, 4_*
    },
    'densenet121_ila_config': {
        **ila_base,
        **densenet121_config,
        'ila_layer': '6',   # choose from 0...8
    },
    'inceptionv3_ila_config': {
        **ila_base,
        **inceptionv3_config,
        'ila_layer': '1_1',     # choose from 1_1, ...
    },
    'inceptionv4_ila_config': {
        **ila_base, 
        **inceptionv4_config,
        'ila_layer': '0',   # chose from 0...21
    },
    'inceptionresnetv2_ila_config': {
        **ila_base,
        **inceptionresnetv2_config,
        'ila_layer': '1_1',     # choose from 1_1, ...
    },
}