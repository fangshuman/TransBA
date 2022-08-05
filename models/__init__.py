from . import cifar10_models
from . import imagenet_models


def get_model_config(dataset, is_source=True):
    if dataset == "CIFAR10":
        return cifar10_models.source_model if is_source else cifar10_models.target_model
    else:
        return imagenet_models.source_model if is_source else imagenet_models.target_model


def make_model(arch, dataset="ImageNet"):
    assert dataset in ["ImageNet", "CIFAR10"]
    if dataset == "ImageNet":
        return imagenet_models.make_model(arch)
    elif dataset == "CIFAR10":
        return cifar10_models.make_model(arch)
