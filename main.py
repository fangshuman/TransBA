import os
import argparse
import logging
import ipdb

import numpy as np
import torch
import torch.nn as nn 
from torchvision import transforms

import configs
from utils import Parameters
from models import make_model
from dataset import make_loader, save_image
from attacks import get_attack



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    #parser.add_argument('--dataset', type=str)
    parser.add_argument('--input-dir', type=str, default='dataset_1000')
    #parser.add_argument('--output-dir', type=str)
    parser.add_argument('--attack-method',type=str, default='i_fgsm', choices=configs.attack_methods)
    #parser.add_argument('--source-model', type=str, default='vgg16', choices=configs.source_model_names)
    #parser.add_argument('--target-model', type=str, choices=target_model_names)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--total-num', type=int, default=1000)
    parser.add_argument('--target', action='store_true', help='targeted attack',)
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--eps', type=float)
    parser.add_argument('--nb-iter', type=int)
    parser.add_argument('--eps-iter', type=float)
    parser.add_argument('--kernlen', type=int, help='for ti-fgsm')
    parser.add_argument('--nsig', type=int, help='for ti-fgsm')
    parser.add_argument('--prob', type=float, help='for di-fgsm')
    parser.add_argument('--decay-factor', type=float, help='for mi-fgsm')
    parser.add_argument('--gamma', type=float, help='for sgm')
    parser.add_argument('--print-freq', type=int, default=1, help='print frequency')
    parser.add_argument('--valid', help='validate adversarial example', action='store_true')
    
    args = parser.parse_args()

    try:
        config = getattr(configs, args.attack_method + '_config')
        args = vars(args)
        args = {**config, **{k: args[k] for k in args if args[k] is not None}}
        args = Parameters(args)
    except Exception:
        raise NotImplementedError(f"No such configuration: {args.config}")

    if not os.path.exists(os.path.join('output', args.attack_method)):
        os.mkdir(os.path.join('output', args.attack_method))
    args.output_dir = os.path.join('output', args.attack_method)

    return args



class Normalize(nn.Module):
    """ Module to normalize an image before entering model """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


def generate_adversarial_example(model, data_loader, attacker, img_list, output_dir):
    model.eval()

    for i, (inputs, labels, indexs) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # generate adversarial example
        inputs_adv = attacker.perturb(inputs, labels) 
        
        # save adversarial example
        save_image(inputs_adv.detach().cpu().numpy(), indexs, 
                   img_list=img_list, output_dir=output_dir)
        

    logger.info(f'Attack finished\n')


def validate(model, val_loader, all_preds):
    model.eval()
    total_num = 0
    suc = 0
    acc = 0

    with torch.no_grad():
        for i, (inputs, labels, indexs) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            clean_preds = torch.from_numpy(all_preds[indexs]).to(device)

            output = model(inputs)
            _, preds = torch.max(output.data, dim=1)
            
            suc += (preds != clean_preds).sum().item()
            acc += (preds == labels).sum().item()
            total_num += inputs.size(0)

    logger.info(f'suc: {suc * 100.0 / total_num:.2f}%\tacc: {acc * 100.0 / total_num:.2f}%\n')



def _main():    
    args = get_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if os.path.exists(os.path.join('output_log', args.attack_method + 'log')):
        os.remove(os.path.join('output_log', args.attack_method))
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join('output_log', args.attack_method + '.log')),
            logging.StreamHandler(),
        ],
    )


    for source_model_name in configs.source_model_names:
        # create model
        model = make_model(arch=source_model_name)
        size = model.input_size[1]
        model = nn.Sequential(Normalize(mean=model.mean, std=model.std), model)
        model = model.to(device)

        # create dataloader
        img_list, data_loader = make_loader(image_dir=args.iuput_dir,
                                            label_dir='imagenet_class_to_idx.npy', 
                                            phase='cln',
                                            batch_size=args.batch_size,
                                            total=args.total_num,
                                            size=size)

        logger.info(f'Attack with {args.attack_method}..')
        attack = get_attack(attack=args.attack_method, 
                            model=model, 
                            loss_fn=nn.CrossEntropyLoss(),
                            args=args)
        generate_adversarial_example(model=model,
                                     data_loader=data_loader,
                                     attacker=attack,
                                     img_list=img_list,
                                     output_dir=args.output_dir)
        
        # validate


    '''
    logger.info(f'Source Model: {args.source_model}\n') 

    # create model
    model = make_model(arch=source_model_name)
    size = model.input_size[1]
    model = nn.Sequential(Normalize(mean=model.mean, std=model.std), model)
    model = model.to(device)

    # create dataloader
    img_list, data_loader = make_loader(image_dir=image_dir,
                                        label_dir=label_dir, 
                                        phase='att',
                                        batch_size=batch_size,
                                        total=total_num,
                                        size=size)

    logger.info(f'Attack with {attack_method}..')
    attacker = Attacker(attack_method=attack_method, 
                        predict=model, 
                        loss_fn=nn.CrossEntropyLoss(),
                        eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                        target=False)
    generate_adversarial_example(model=model, data_loader=data_loader, 
                                 attacker=attacker, 
                                 img_list=img_list, output_dir=output_dir)
    
    # validate
    if args.valid:
        for model_name in model_names:
            t_model = make_model(model_name)
            t_size = t_model.input_size[1]
            t_model = nn.Sequential(Normalize(mean=t_model.mean, std=t_model.std), t_model)
            t_model = t_model.to(device)

            all_preds = np.load(os.path.join('./model_preds', model_name + '.npy'),allow_pickle=True)[()]

            _, val_loader = make_loader(root_dir=output_dir,
                                        phase='val',
                                        batch_size=eval_batch_size[model_name],
                                        total=args.total_num,
                                        size=t_size)
            
            logger.info(f'Transfer to {model_name}...')
            validate(t_model, val_loader, all_preds)
    '''


if __name__ == '__main__':
    main()