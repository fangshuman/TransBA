import torch
import torch.nn as nn
import torch.nn.functional as F


class Parameters:
    def __init__(self, params):
        for k, v in params.items():
            self.__setattr__(k, v)

    def __setattr__(self, key, value):
        self.__dict__[key.lower()] = value

    def __repr__(self):
        return repr(self.__dict__)


