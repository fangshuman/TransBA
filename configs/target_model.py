# target model
target_model_names = [
    'vgg16_bn',
    'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet169', 'densenet201',
    'inceptionv3', 'inceptionv4', 'inceptionresnetv2',
    #'robust_models',
]


# target model batch size
target_model_batch_size = {
    'vgg16_bn': 100, 
    'resnet50': 250, 'resnet101': 250, 'resnet152': 250,
    'densenet121': 250, 'densenet169': 250, 'densenet201': 250,
    'inceptionv3': 250, 'inceptionv4': 250, 'inceptionresnetv2': 250,
}
