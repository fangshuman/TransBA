# source model
source_model_names = [
    'vgg16',
    'resnet50',
    'densenet121',
    'inceptionv3',
    'inceptionv4',
    'inceptionresnetv2',
]


# source model batch size
att_batch_size = {
    'vgg16': 32,
    'resnet50': 32,
    'densenet121': 32,
    'inceptionv3': 32,
    'inceptionv4': 16,
    'inceptionresnetv2': 16,
}



'''
vgg16_config = {
    'source_model': 'vgg16',
    'batch_size': 32,
}

resnet50_config = {
    'source_model': 'resnet50',
    'batch_size': 64,
}

densenet121_config = {
    'source_model': 'densenet121',
    'batch_size': 64,
}

inceptionv3_config = {
    'source_model': 'inceptionv3',
    'batch_size': 32,
}

inceptionv4_config = {
    'source_model': 'inceptionv4',
    'batch_size': 32,
}

inceptionresnetv2_config = {
    'source_model': 'inceptionresnetv2',
    'batch_size': 32,
}
'''