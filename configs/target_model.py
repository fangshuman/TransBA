# target model
target_model_names = [
    'vgg16_bn',
    'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet169', 'densenet201',
    'inceptionv3', 'inceptionv4', 'inceptionresnetv2',
]


# target model batch size
target_model_batch_size = {
    'vgg16_bn': 128, 
    'resnet50': 256, 'resnet101': 128, 'resnet152': 64,
    'densenet121': 256, 'densenet169': 128, 'densenet201': 128,
    'inceptionv3': 128, 'inceptionv4': 128, 'inceptionresnetv2': 128,
}
