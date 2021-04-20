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
    parser.add_argument('--attack-method',type=str, choices=configs.attack_methods)
    #parser.add_argument('--source-model', type=str, default='vgg16', choices=configs.source_model_names)
    #parser.add_argument('--target-model', type=str, choices=target_model_names)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--total-num', type=int, default=1000)
    parser.add_argument('--target', action='store_true', help='targeted attack',)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--nb-iter', type=int)
    parser.add_argument('--eps-iter', type=float)
    parser.add_argument('--kernlen', type=int, help='for ti-fgsm')
    parser.add_argument('--nsig', type=int, help='for ti-fgsm')
    parser.add_argument('--prob', type=float, help='for di-fgsm')
    parser.add_argument('--decay-factor', type=float, help='for mi-fgsm')
    parser.add_argument('--gamma', type=float, help='for sgm gamma < 1.0')
    parser.add_argument('--print-freq', type=int, default=4, help='print frequency')
    parser.add_argument('--not-valid', help='validate adversarial example', action='store_true')
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




def predict(model, data_loader):
    model.eval()
    total_num = 0
    count = 0
    preds_ls = []

    with torch.no_grad():
        for i, (inputs, labels, indexs) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = model(inputs)
            _, preds = torch.max(output.data, dim=1)
            preds_ls.append(preds)
            
            total_num += inputs.size(0)
            count += (preds == labels).sum().item()
    
    return count, total_num, preds_ls


def attack_source_model(arch, args):
    # create model
    model = make_model(arch=arch)
    size = model.input_size[1]
    model = model.cuda()

    # create dataloader
    img_list, data_loader = make_loader(image_dir=args.input_dir,
                                        label_dir='imagenet_class_to_idx.npy', 
                                        phase='cln',
                                        batch_size=configs.att_batch_size[arch],
                                        total=args.total_num,
                                        size=size)

    # create attack
    attack = get_attack(attack=args.attack_method, 
                        model_name=arch,
                        model=model, 
                        loss_fn=nn.CrossEntropyLoss(),
                        args=args)

    # generate adversarial example
    model.eval()
    if args.gamma < 1.0:  # use Skip Gradient Method (SGM)
        attack.register_hook()
    for i, (inputs, _, indexs) in enumerate(data_loader):
        inputs = inputs.cuda()
            
        output = model(inputs)
        _, labels = torch.max(output, dim=1)

        inputs_adv = attack.perturb(inputs, labels) 
            
        # save adversarial example
        if not os.path.exists(os.path.join(args.output_dir, arch)):
            os.mkdir(os.path.join(args.output_dir, arch))
        save_image(images=inputs_adv.detach().cpu().numpy(), 
                   indexs=indexs, 
                   img_list=img_list, 
                   output_dir=os.path.join(args.output_dir, arch))

        if i % args.print_freq == 0:
            print(f'generating: [{i} / {len(data_loader)}]')
    

def predict_model_with_clean_example(arch, args):
    model = make_model(arch=arch)
    size = model.input_size[1]
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


def valid_model_with_adversarial_example(source_arch, target_arch, args):
    model = make_model(arch=target_arch)
    size = model.input_size[1]
    model = model.cuda()

    _, data_loader = make_loader(#image_dir=os.path.join(args.output_dir, source_arch),
                                 image_dir='../trans/skip-connections-matter-master/adv_images_resnet50',                          
                                 label_dir=os.path.join('output_clean_preds', target_arch + '.npy'), 
                                 phase='adv',
                                 batch_size=configs.val_batch_size[target_arch], 
                                 total=args.total_num,
                                 size=size)
    
    cnt, total, _ = predict(model, data_loader)
    return (total - cnt) * 100.0 / total


def main():    
    args = get_args()
    
    if os.path.exists(os.path.join('output_log', args.attack_method + '.log')):
        os.remove(os.path.join('output_log', args.attack_method + '.log'))
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            #logging.FileHandler(os.path.join('output_log', args.attack_method + '.log')),
            logging.FileHandler(os.path.join('output_log', 'sgm_resnet50.log')),
            logging.StreamHandler(),
        ],
    )
    '''
    if args.clean_pred:
        for target_model_name in configs.target_model_names:
            logger.info(f'Predict {target_model_name} with clean example..')
            predict_model_with_clean_example(target_model_name, args)
            logger.info('Predict done.')
        
    logger.info(f'Generate adversarial examples with {args.attack_method}')
    
    for i, source_model_name in enumerate(configs.source_model_names):
        logger.info('-' * 50)
        logger.info(f'[{i+1} / {len(configs.source_model_names)}] source model: {source_model_name}')

        attack_source_model(source_model_name, args)
        logger.info(f'Attack finished.')
        
        # validate
        if not args.not_valid:
            for target_model_name in configs.target_model_names:
                logger.info(f'Transfer to {target_model_name}..')
                succ_rate = valid_model_with_adversarial_example(source_model_name, target_model_name, args)
                logger.info(f'succ rate: {succ_rate:.2f}%')
                logger.info(f'Transfer done.')
    '''
    for target_model_name in configs.target_model_names:
        logger.info(f'Transfer to {target_model_name}..')
        succ_rate = valid_model_with_adversarial_example('', target_model_name, args)
        logger.info(f'succ rate: {succ_rate:.2f}%')
        logger.info(f'Transfer done.')

        

            


if __name__ == '__main__':
    main()