source_model = {
    "resnet50": {"source_model": "resnet50", "batch_size": 1000},
    "densenet121": {"source_model": "densenet121", "batch_size": 500},
}

target_model = {
    "googlenet": 1000,
    "vgg11_bn": 1000,
    "vgg13_bn": 1000,
    "vgg16_bn": 1000,
    "vgg19_bn": 1000,
    "resnet18": 2000,
    "resnet34": 2000,
    "resnet50": 2000,
    "densenet121": 2000,
    "densenet169": 2000,
    "inceptionv3": 2000,
    "mobilenetv2": 2000,
}
