import os
import ipdb

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import configs
from models import make_model
from dataset import make_loader


def predict(model, data_loader):
    model.eval()
    total_num = 0
    count = 0

    with torch.no_grad():
        for i, (inputs, labels, indexs) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = model(inputs)
            _, preds = torch.max(output.data, dim=1)
            
            total_num += inputs.size(0)
            count += (preds == labels).sum().item()
    
    return count * 100.0 / total_num


def eval_all_iters(root_dir, arch):
    model = make_model(arch=arch)
    size = model.input_size[1]
    model = model.cuda()

    iters = []
    accls = []
    for i in range(10, 310, 10):
        print(f'evaluate {i}iter adv images')
        _, data_loader = make_loader(image_dir=os.path.join(root_dir, str(i) + 'iter'),
                                     label_dir='imagenet_class_to_idx.npy',
                                     phase='adv',
                                     batch_size=configs.val_batch_size[arch],
                                     total=1000,
                                     size=size)
        acc = predict(model, data_loader)
        print(acc)
        iters.append(i)
        accls.append(acc)
    return iters, accls
        


def main():
    root_dir = './output/mifgsm_resnet50'

    for target_model in configs.target_model_names:
        print(target_model)
        iters, accls = eval_all_iters(root_dir, target_model)    
        plt.plot(iters, accls, ls='-', label=target_model)
        plt.legend()
        plt.savefig(f'figs/mifgsm/{target_model}.png')
        plt.clf()
        plt.cla()
    


if __name__ == '__main__':
    main()
    