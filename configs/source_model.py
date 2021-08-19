# source model
ImageNet_source_model = {
    'vgg16_bn': {'source_model': 'vgg16_bn', 'batch_size': 32},
    'resnet50': {'source_model': 'resnet50', 'batch_size': 80},
    'resnet152': {'source_model': 'resnet152', 'batch_size': 50},
    'densenet121': {'source_model': 'densenet121', 'batch_size': 32},
    'densenet201': {'source_model': 'densenet201', 'batch_size': 32},
    'inceptionv3': {'source_model': 'inceptionv3', 'batch_size': 64},
    'inceptionv4': {'source_model': 'inceptionv4', 'batch_size': 32},
    'inceptionresnetv2': {'source_model': 'inceptionresnetv2', 'batch_size': 32},
}

CIFAR10_source_model = {
    'resnet50': {'source_model': 'resnet50', 'batch_size': 1000},
    'densenet121': {'source_model': 'densenet121', 'batch_size': 500},
}

def get_source_model_config(set):
    if set == "CIFAR10":
        return CIFAR10_source_model
    else:
        return ImageNet_source_model

