import os
import ipdb

import numpy as np
import torch
import torch.nn as nn 
from torchvision import transforms
from datetime import datetime

from get_model import make_model
from dataset import make_loader


model_names = [
    'vgg16',
    'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet161', 'densenet201',
    'inceptionv3', 'inceptionv4', 'inceptionresnetv2',
]


batch_size = {
    'vgg16': 128,
    'resnet50': 256, 'resnet101': 128, 'resnet152': 64,
    'densenet121': 256, 'densenet161': 128, 'densenet201': 128,
    'inceptionv3': 128, 'inceptionv4': 128, 'inceptionresnetv2': 128,
}



class Normalize(nn.Module):
    """ Module to normalize an image before entering model """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


def main():
    device = 'cuda'

    for model_name in model_names:
        print(f'predict with {model_name}...')
        model = make_model(arch=model_name)
        size = model.input_size[1]
        model = nn.Sequential(Normalize(mean=model.mean, std=model.std), model)
        model = model.to(device)

        img_list, data_loader = make_loader(root_dir='./dataset_1000',
                                            phase='att',
                                            batch_size=batch_size[model_name],
                                            total=1000,
                                            size=size)

        model.eval()
        all_preds = []
        with torch.no_grad():
            for i, (inputs, labels, indexs) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                _, preds = torch.max(output.data, dim=1)

                all_preds.append(preds)
        
        all_preds = torch.cat(all_preds).detach().cpu().numpy()
        np.save(os.path.join('model_preds', model_name + '.npy'), all_preds)
        print('predict done.')



if __name__ == '__main__':
    main()