# target model
ImageNet_target_model = {
    'vgg16_bn': 100, 
    'resnet50': 250, 'resnet101': 250, 'resnet152': 250,
    'densenet121': 250, 'densenet169': 250, 'densenet201': 250,
    'inceptionv3': 250, 'inceptionv4': 250, 'inceptionresnetv2': 250,
}

CIFAR10_target_model = {
    'googlenet': 1000,
    'vgg11_bn': 1000, 'vgg13_bn': 1000, 'vgg16_bn': 1000, 'vgg19_bn': 1000,
    'resnet18': 2000, 'resnet34': 2000, 'resnet50': 2000,
    'densenet121': 2000, 'densenet169': 2000,
    'inceptionv3': 2000,
    'mobilenetv2': 2000,
}

def get_target_model_config(set):
    if set == "CIFAR10":
        return CIFAR10_target_model
    else:
        return ImageNet_target_model