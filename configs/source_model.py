# source model
source_model_names = [
    'vgg16_bn',
    'resnet50',
    'resnet152',
    'densenet121',
    'densenet201',
    'inceptionv3',
    'inceptionv4',
    'inceptionresnetv2',
]


# target model config
vgg16_bn_config = {
    'source_model': 'vgg16_bn',
    'batch_size': 32,
}

resnet50_config = {
    'source_model': 'resnet50',
    'batch_size': 80,
}

resnet152_config = {
    'source_model': 'resnet152',
    'batch_size': 50,
}

densenet121_config = {
    'source_model': 'densenet121',
    'batch_size': 32,
}

densenet201_config = {
    'source_model': 'densenet201',
    'batch_size': 32,
}

inceptionv3_config = {
    'source_model': 'inceptionv3',
    'batch_size': 64,
}

inceptionv4_config = {
    'source_model': 'inceptionv4',
    'batch_size': 32,
}

inceptionresnetv2_config = {
    'source_model': 'inceptionresnetv2',
    'batch_size': 32,
}
