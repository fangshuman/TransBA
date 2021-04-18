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
from evaluate import predict



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    #parser.add_argument('--dataset', type=str)
    parser.add_argument('--input-dir', type=str, default='dataset_1000')
    #parser.add_argument('--output-dir', type=str)
    parser.add_argument('--attack-method',type=str, choices=configs.attack_methods)
    #parser.add_argument('--source-model', type=str, default='vgg16', choices=configs.source_model_names)
    #parser.add_argument('--target-model', type=str, choices=target_model_names)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--total-num', type=int, default=1000)
    parser.add_argument('--target', action='store_true', help='targeted attack',)
    #parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
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
    parser.add_argument('--clean-pred', help='get prediction of clean examples', action='store_true')
    
    args = parser.parse_args()

    try:
        config = getattr(configs, args.config)
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


def attack_source_model(arch, args):
    # create model
    model = make_model(arch=arch)
    size = model.input_size[1]
    model = nn.Sequential(Normalize(mean=model.mean, std=model.std), model)
    model = model.cuda()

    # create dataloader
    img_list, data_loader = make_loader(image_dir=args.iuput_dir,
                                        label_dir='imagenet_class_to_idx.npy', 
                                        phase='cln',
                                        batch_size=configs.att_batch_size[arch],
                                        total=args.total_num,
                                        size=size)

    # create attack
    attack = get_attack(attack=args.attack_method, 
                        model=model, 
                        loss_fn=nn.CrossEntropyLoss(),
                        args=args)

    # generate adversarial example
    model.eval()
    for i, (inputs, _, indexs) in enumerate(data_loader):
        inputs = inputs.cuda()
            
        output = model(inputs)
        _, labels = torch.max(output, dim=1).item()

        inputs_adv = attack.perturb(inputs, labels) 
            
        # save adversarial example
        save_image(images=inputs_adv.detach().cpu().numpy(), 
                    indexs=indexs, 
                    img_list=img_list, 
                    output_dir=os.path.join(args.output_dir, arch))

        if i % args.print_freq == 0:
            print(f'generating: [{i} / {len(data_loader)}]')
    

def predict_model_with_clean_example(arch, args):
    model = make_model(arch=arch)
    size = model.input_size[1]
    model = nn.Sequential(Normalize(mean=model.mean, std=model.std), model)
    model = model.cuda()

    _, data_loader = make_loader(image_dir=args.input_dir,
                                 label_dir='imagenet_class_to_idx.npy', 
                                 phase='cln',
                                 batch_size=configs.val_batch_size[arch],
                                 total=args.total_num,
                                 size=size)
    
    _, _, preds_ls = predict(model, data_loader)
    cln_preds = torch.cat(preds_ls).cpu().numpy()
    np.save(os.path.join('output_clean_preds', arch + '.npy'), cln_preds)


def valid_model_with_adversarial_example(arch, args):
    model = make_model(arch=arch)
    size = model.input_size[1]
    model = nn.Sequential(Normalize(mean=model.mean, std=model.std), model)
    model = model.cuda()

    _, data_loader = make_loader(image_dir=os.path.join(args.output_dir),
                                 label_dir=os.path.join('output_clean_preds', arch + '.npy'), 
                                 phase='att',
                                 batch_size=configs.val_batch_size[arch], 
                                 total=args.total_num,
                                 size=size)
    
    cnt, total, _ = predict(model, data_loader)
    return (1 - cnt) * 100.0 / total


def main():    
    args = get_args()

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

    if args.clean_pred:
        for target_model_name in configs.target_model_names:
            logger.info(f'Predict {target_model_name} with clean example..')
            predict_model_with_clean_example(target_model_name, args)
            logger.info('Predict done.')

    logger.info('Begin to generate adversarial examples in all source models.\n')

    ipdb.set_trace()

    for i, source_model_name in enumerate(configs.source_model_names):
        logger.info(f'[{i} / {len(configs.source_model_names)}] source model: {source_model_name}')

        logger.info(f'Attack with {args.attack_method}..')
        attack_source_model(source_model_name, args)
        logger.info(f'Attack finished.\n')
        
        # validate
        if args.valid:
            for target_model_name in configs.target_model_names:
                logger.info(f'Transfer to {target_model_name}..')
                succ_rate = valid_model_with_adversarial_example(target_model_name, args)
                logger.info(f'succ rate: {succ_rate:.2f}')
                logger.info(f'Transfer done.')


if __name__ == '__main__':
    main()