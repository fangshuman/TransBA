import argparse
import ipdb

import torch
import torch.nn as nn
from datetime import datetime

from get_model import make_model
from dataset import make_loader


model_names = [
    'vgg16',
    'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet161', 'densenet201',
    'inceptionv3', 'inceptionv4', 'inceptionresnetv2',
]


parser = argparse.ArgumentParser(description='Pytorch ImageNet Untargeted Attack Evaluate')
parser.add_argument('--input-dir', help='path to dataset')
parser.add_argument('--total-num', type=int, default=1000,
                    help='number of images to attack')
parser.add_argument('--arch', default='vgg16', help='target model', choices=model_names)
parser.add_argument('--batch-size', type=int, default=64, 
                    help='input batch size for adversarial attack')
parser.add_argument('--targeted', help='targeted attack', action='store_true')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--print-freq', type=int, default=1, help='print frequency')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# add Module to normalize an image before entering model
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


def main():
    # print args
    timestamp = str(datetime.now())[:-7]
    print(f'{timestamp}\n'
          f'source:  {args.input_dir}\n'
          f'model:   {args.arch}\n'
          f'dataset: ImageNet\n')

    # create model
    model = make_model(arch=args.arch)
    input_size = model.input_size[1]
    model = nn.Sequential(Normalize(mean=model.mean, std=model.std), model)
    model = model.to(device)

    # create dataloader
    _, data_loader = make_loader(root_dir=args.input_dir, 
                                 phase='val',
                                 batch_size=args.batch_size,
                                 total=args.total_num,
                                 size=input_size)

    # validate
    validate(model, data_loader, total_num=args.total_num)


def validate(model, data_loader, total_num=1000):
    model.eval()
    correct = 0

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            _, preds = torch.max(output, dim=1)
            correct += (preds == labels).sum().item()

            # print
            if i % args.print_freq == 0:
                print(f'evaluating: [{i} / {len(data_loader)}]')

        timestamp = str(datetime.now())[:-7]
        print(f'\n{timestamp}')
        print(f'Evaluate finished.')
        print(f'acc: {correct * 100.0 / total_num: .2f}%')



if __name__ == '__main__':
    main()