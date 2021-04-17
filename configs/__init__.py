from .source_model import *
from .attack_method import *


source_model_names = [
    'vgg16',
    'resnet50',
    'densenet121',
    'inceptionv3',
    'inceptionv4',
    'inceptionresnetv2'
]
target_model_names = [
    'vgg16',
    'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet169', 'densenet201',
    'inceptionv3', 'inceptionv4', 'inceptionresnetv2',
]

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