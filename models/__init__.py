import torch
import torchvision.models as thmodels
import pretrainedmodels


def make_model(arch):
    if arch == 'vgg16':
        model = vgg16()
    elif arch == 'resnet50':
        model = resnet50()
    elif arch == 'resnet101':
        model = resnet101()
    elif arch == 'resnet152':
        model = resnet152()
    elif arch == 'densenet121':
        model = densenet121()
    elif arch == 'densenet161':
        model = densenet161()
    elif arch == 'densenet201':
        model = densenet201()
    elif arch == 'inceptionv3':
        model = inceptionv3()
    elif arch == 'inceptionv4':
        model = inceptionv4()
    elif arch == 'inceptionresnetv2':
        model = inceptionresnetv2()
    else:
        raise NotImplementedError(f"No such model: {arch}")
    
    # for inception* networks
    # model.input_size=[3, 299, 299]
    # model.mean=[0.5, 0.5, 0.5]
    # model.std =[0.5, 0.5, 0.5]

    # for resnet* networks
    # model.input_size=[3, 224, 224]
    # model.mean=[0.485, 0.456, 0.406]
    # model.std =[0.229, 0.224, 0.225]                          

    return model


def vgg16():
    return pretrainedmodels.vgg16_bn(num_classes=1000, pretrained='imagenet')

def resnet50():
    return pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet')

def resnet101():
    return pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet')

def resnet152():
    return pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet')

def densenet121():
    return pretrainedmodels.densenet121(num_classes=1000, pretrained='imagenet')

def densenet161():
    return pretrainedmodels.densenet161(num_classes=1000, pretrained='imagenet')

def densenet201():
    return pretrainedmodels.densenet201(num_classes=1000, pretrained='imagenet')

def inceptionv3():
    return pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet')

def inceptionv4():
    return pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet')

def inceptionresnetv2():
    return pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet')

