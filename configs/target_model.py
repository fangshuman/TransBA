# target model
target_model_names = [
    'vgg16_bn',
    'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet169', 'densenet201',
    'inceptionv3', 'inceptionv4', 'inceptionresnetv2',
    'robust_models'
]


# target model batch size
target_model_batch_size = {
    'vgg16_bn': 100, 
    'resnet50': 500, 'resnet101': 500, 'resnet152': 500,
    'densenet121': 500, 'densenet169': 500, 'densenet201': 500,
    'inceptionv3': 250, 'inceptionv4': 250, 'inceptionresnetv2': 250,
}
