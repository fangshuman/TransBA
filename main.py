import argparse
import os
import ipdb

import torch
import torch.nn as nn 
from torchvision import transforms
from datetime import datetime

from get_model import make_model
from dataset import make_loader, save_image
from attacker import Attacker


model_names = [
    'vgg16',
    'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet161', 'densenet201',
    'inceptionv3', 'inceptionv4', 'inceptionresnetv2',
]

attack_methods = [
    'i-fgsm',
    'ti-fgsm',
    'di-fgsm',
    'mi-fgsm',
    'si-fgsm',
    'admix', 
    'emi-fgsm', 
    'vi-fgsm', 
    'pi-fgsm', 
    'SGM',
    'LinBP',
]

eval_batch_size = {
    'vgg16': 128,
    'resnet50': 256, 'resnet101': 128, 'resnet152': 64,
    'densenet121': 256, 'densenet161': 128, 'densenet201': 128,
    'inceptionv3': 128, 'inceptionv4': 128, 'inceptionresnetv2': 128,
}


parser = argparse.ArgumentParser(description='Pytorch ImageNet Untargeted Attack')
parser.add_argument('--input-dir', default='./dataset_1000', 
                    help='input root directory')
parser.add_argument('--output-dir', default='', help='output root directory')
parser.add_argument('--total-num', type=int, default=1000,
                    help='number of images to attack')
parser.add_argument('--arch', default='vgg16', help='source model', choices=model_names)
parser.add_argument('--batch-size', type=int, default=64, 
                    help='input batch size for adversarial attack')
parser.add_argument('--targeted', help='targeted attack', action='store_true')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--attack-method', default='i-fgsm', choices=attack_methods,
                    help='attack method')
parser.add_argument('--epsilon', type=float, default=16, help='perturbation')
parser.add_argument('--num-steps', type=int, default=10, help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=-1, 
                    help='perturb step size, equivalent to epsilon/num_steps when setting -1')
parser.add_argument('--print-freq', type=int, default=1, help='print frequency')
parser.add_argument('--valid', help='validate adversarial example', action='store_true')

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


# generate and save adversarial example
def generate_adversarial_example(model, data_loader, attacker, img_list, output_dir, total_num=1000):
    model.eval()
    ori_correct = 0
    #adv_correct = 0

    for i, (inputs, labels, indexs) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(inputs)
            _, preds = torch.max(output, dim=1)
            ori_correct += (preds == labels).sum().item()
        
        # generate adversarial example
        inputs_adv = attacker.perturb(inputs, labels) 

        #with torch.no_grad():
        #    output = model(inputs_adv)
        #    _, preds = torch.max(output, dim=1)
        #    adv_correct += (preds == labels).sum().item()
        
        # save adversarial example
        save_image(inputs_adv.detach().cpu().numpy(), indexs, 
                   img_list=img_list, output_dir=output_dir)
        
        # print
        if i % args.print_freq == 0:
            print(f'generating: [{i} / {len(data_loader)}]')

    timestamp = str(datetime.now())[:-7]
    print(f'\nAttack finished at {timestamp}')
    print(f'clean acc:  {ori_correct * 100.0 / total_num: .2f}%\n')
    #print(f'attack acc: {adv_correct * 100.0 / total_num: .2f}%')


# validate adversarial example
def validate(model, val_loader, total_num=1000):
    model.eval()
    correct = 0

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            _, preds = torch.max(output, dim=1)
            correct += (preds == labels).sum().item()

        timestamp = str(datetime.now())[:-7]
        print(f'Evaluate finished at {timestamp}')
        print(f'acc: {correct * 100.0 / total_num: .2f}%\n')


def main():
    
    # print args
    timestamp = str(datetime.now())[:-7]
    print(f'{timestamp}\nattacker:  {args.attack_method}\n'
          f'model:     {args.arch}\n'
          f'dataset:   ImageNet\n'
          f'eps:       {args.epsilon}\n'
          f'n_iter:    {args.num_steps}\n'
          f'step_size: {args.step_size if args.step_size > 0 else args.epsilon/args.num_steps}\n')

    # create output directory
    if args.output_dir == '':
        output_dir = 'output/' + args.attack_method + '-' + args.arch
    else:
        output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # create model
    model = make_model(arch=args.arch)
    size = model.input_size[1]
    model = nn.Sequential(Normalize(mean=model.mean, std=model.std), model)
    model = model.to(device)

    # create dataloader
    img_list, data_loader = make_loader(root_dir=args.input_dir, 
                                        phase='att',
                                        batch_size=args.batch_size,
                                        total=args.total_num,
                                        size=size)

    # create adversarial
    epsilon = args.epsilon / 255.0
    if args.step_size < 0:
        step_size = epsilon / args.num_steps
    else:
        step_size = args.step_size / 255.0
    
    attacker = Attacker(args.attack_method, 
                        predict=model, 
                        loss_fn=nn.CrossEntropyLoss(),
                        eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                        targeted=False)
    generate_adversarial_example(model=model, data_loader=data_loader, 
                                 attacker=attacker, 
                                 img_list=img_list, output_dir=output_dir,
                                 total_num=args.total_num)
    
    # validate
    if args.valid:
        for model_name in model_names:
            t_model = make_model(model_name)
            t_size = t_model.input_size[1]
            t_model = nn.Sequential(Normalize(mean=t_model.mean, std=t_model.std), t_model)
            t_model = t_model.to(device)

            _, val_loader = make_loader(root_dir=output_dir,
                                        phase='val',
                                        batch_size=eval_batch_size[model_name],
                                        total=args.total_num,
                                        size=t_size)
            
            print(f'attack {model_name}...')
            validate(t_model, val_loader, total_num=args.total_num)


if __name__ == '__main__':
    main()