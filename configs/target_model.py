# target model
target_model_names = [
    'vgg16',
    'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet169', 'densenet201',
    'inceptionv3', 'inceptionv4', 'inceptionresnetv2',
]


# target model batch size
val_batch_size = {
    'vgg16': 128, 'vgg19': 128,
    'resnet50': 256, 'resnet101': 128, 'resnet152': 64,
    'densenet121': 256, 'densenet169': 128, 'densenet201': 128,
    'inceptionv3': 128, 'inceptionv4': 128, 'inceptionresnetv2': 128,
}