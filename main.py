import os
import argparse
import logging
from datetime import datetime
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

import configs
from utils import Parameters
from models import make_model
from dataset import make_loader, save_image
from attacks import get_attack

seed = 0
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)  # as reproducibility docs
torch.manual_seed(seed)  # as reproducibility docs
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False  # as reproducibility docs
torch.backends.cudnn.deterministic = True  # as reproducibility docs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="dataset_1000")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument(
        "--attack-method", type=str, default="i_fgsm", choices=configs.attack_methods
    )
    # parser.add_argument('--source-model', type=str, default='vgg16', choices=configs.source_model_names)
    # parser.add_argument('--target-model', type=str, choices=target_model_names)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--total-num", type=int, default=1000)
    parser.add_argument(
        "--target",
        action="store_true",
        help="targeted attack",
    )
    parser.add_argument("--eps", type=float)
    parser.add_argument("--nb-iter", type=int)
    parser.add_argument("--eps-iter", type=float)
    parser.add_argument("--kernlen", type=int, help="for ti-fgsm")
    parser.add_argument("--nsig", type=int, help="for ti-fgsm")
    parser.add_argument("--prob", type=float, help="for di-fgsm")
    parser.add_argument("--decay-factor", type=float, help="for mi-fgsm")
    parser.add_argument("--gamma", type=float, help="for sgm gamma < 1.0")
    parser.add_argument("--ila-layer", type=int, help="for ila")
    parser.add_argument("--step_size_pgd", type=float, help="for ila")
    parser.add_argument("--step_size_ila", type=float, help="for ila")
    parser.add_argument("--print-freq", type=float, default=10, help="print frequency")
    parser.add_argument(
        "--not-valid", help="validate adversarial example", action="store_true"
    )

    args = parser.parse_args()

    return args


def predict(model, data_loader):
    model.eval()
    total_num = 0
    count = 0
    # preds_ls = []

    with torch.no_grad():
        for i, (inputs, labels, indexs) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = model(inputs)
            _, preds = torch.max(output.data, dim=1)
            # preds_ls.append(preds.cpu())

            total_num += inputs.size(0)
            count += (preds == labels).sum().item()

    # return count, total_num, preds_ls
    return count, total_num


def attack_source_model(arch, args, is_to_save=True):
    # create model
    model = make_model(arch=arch)
    size = model.input_size[1]
    model = model.cuda()

    # create dataloader
    batch_size = int(args.batch_size * args.batch_size_coeff)
    img_list, data_loader = make_loader(
        image_dir=args.input_dir,
        label_dir="imagenet_class_to_idx.npy",
        phase="cln",
        batch_size=batch_size,
        total=args.total_num,
        size=size,
    )

    # create attack
    attack = get_attack(
        attack=args.attack_method, model=model, loss_fn=F.cross_entropy, args=args
    )

    _advs = []
    _labels = []

    # generate adversarial example
    model.eval()
    if args.gamma < 1.0:  # use Skip Gradient Method (SGM)
        attack.register_hook()
    for i, (inputs, labels, indexs) in enumerate(data_loader):
        _labels.append(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()

        inputs_adv = attack.perturb(inputs, labels)

        # save adversarial example
        if is_to_save:
            save_image(
                images=inputs_adv.detach().cpu().numpy(),
                indexs=indexs,
                img_list=img_list,
                output_dir=args.output_dir,
            )
        else:
            _advs.append(inputs_adv.detach().cpu())

        if i % args.print_freq == 0:
            print(f"generating: [{i} / {len(data_loader)}]")
    return _advs, _labels


def valid_model_with_adversarial_example(arch, args, _advs=None, _labels=None):
    model = make_model(arch=arch)
    size = model.input_size[1]
    model = model.cuda()
    model.eval()
    total = 0
    count = 0

    if _advs is not None and len(_advs) > 0:
        with torch.no_grad():
            for (inputs, labels) in zip(_advs, _labels):
                inputs = inputs.cuda()
                labels = labels.cuda()

                output = model(inputs)
                _, preds = torch.max(output.data, dim=1)

                total += inputs.size(0)
                count += (preds == labels).sum().item()

        return count * 100.0 / total

    _, data_loader = make_loader(
        image_dir=args.output_dir,
        label_dir=os.path.join("imagenet_class_to_idx.npy"),
        phase="adv",
        batch_size=configs.val_batch_size[arch],
        total=args.total_num,
        size=size,
    )

    with torch.no_grad():
        for i, (inputs, labels, indexs) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = model(inputs)
            _, preds = torch.max(output.data, dim=1)

            total += inputs.size(0)
            count += (preds == labels).sum().item()

    return count * 100.0 / total


def main():
    _args = get_args()
    print(_args)
    white_arguments = ['attack_method', 'total_num', 'target', 'eps', 'nb_iter', 'eps_iter']
    black_arguments = ['input_dir', 'output_dir', 'print_freq', 'not_valid']
    log_name = []
    for a in white_arguments:
        v = getattr(_args, a)
        if v is not None:
            log_name.append(f"{a}_{v}")
    for k, v in vars(_args).items():
        if k in white_arguments or k in black_arguments:
            continue
        if v is not None:
            log_name.append(f"{k}_{v}")

    log_name = '-'.join(log_name)

    output_root_dir = os.path.join(_args.output_dir, log_name)
    logger_path = os.path.join(
        "output_log", log_name + ".log"
    )

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    os.makedirs("output_log", exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logger_path),
            logging.StreamHandler(),
        ],
    )

    all_configs = getattr(configs, _args.attack_method + "_config")

    # generate adversarial examples
    logger.info(f"Generate adversarial examples with {_args.attack_method}")
    for i, source_model_name in enumerate(configs.source_model_names):
        logger.info(f"Attacking {source_model_name}...")
        acc_list = []

        if _args.attack_method == "sgm" and source_model_name not in [
            "resnet50",
            "densenet121",
        ]:
            continue

        config_str = source_model_name + "_" + _args.attack_method + "_config"

        if config_str in all_configs:
            config = all_configs[config_str]

            # make output dir
            output_dir = os.path.join(output_root_dir, source_model_name)
            # output_dir = os.path.join(output_root_dir, source_model_name + '_gamma' + str(_args.gamma))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            _args.output_dir = output_dir

            # load config
            args = vars(_args)
            args = {**config, **{k: args[k] for k in args if args[k] is not None}}
            args = Parameters(args)

            logger.info(str(datetime.now())[:-7])
            logger.info(args)
            logger.info(args.nb_iter)

            # begin attack
            logger.info(
                f"[{i+1} / {len(configs.source_model_names)}] source model: {source_model_name}"
            )
            attack_source_model(source_model_name, args)
            logger.info(f"Attack finished.")

            # validate
            if not args.not_valid:
                for target_model_name in configs.target_model_names:
                    logger.info(f"Transfer to {target_model_name}..")
                    acc = valid_model_with_adversarial_example(target_model_name, args)
                    acc_list.append(acc)
                    logger.info(f"acc: {acc:.2f}%")
                    logger.info(f"Transfer done.")

        torch.cuda.empty_cache()

        logger.info("\t".join([str(round(v, 2)) for v in acc_list]))


if __name__ == "__main__":
    main()
