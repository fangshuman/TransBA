import importlib
import os
import argparse
import logging
import random
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import functional as F

import configs
from utils import Parameters
from models import make_model
from dataset import make_loader, save_image
from attacks import get_attack
from evaluate import evaluate_with_natural_model
from eval_robust_models import evaluate_with_robust_model


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
    parser.add_argument("--dataset", type=str, default="ImageNet", choices=["ImageNet", "CIFAR10"])
    parser.add_argument("--input-dir", type=str, default="dataset_1000")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--attack-method", type=str, default="i_fgsm")
    parser.add_argument("--source-model", nargs="+", default="")
    parser.add_argument('--target-model', nargs="+", default="")
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
    parser.add_argument(
        "--not-valid", help="validate adversarial example", action="store_true"
    )
    parser.add_argument(
        "--inputfolder", help="input dataset in folder dataset", action="store_true"
    )

    args = parser.parse_args()

    return args



def attack_source_model(arch, args):
    # create model
    model = make_model(arch=arch, dataset=args.dataset)
    size = model.input_size[1]
    model = model.cuda()

    if args.inputfolder:
        label_dir = None
    else:
        label_dir = "TrueLabel.npy" if args.dataset == "ImageNet" else "cifar10_class_to_idx.npy"

    # create dataloader
    batch_size = args.batch_size
    img_list, data_loader = make_loader(
        image_dir=args.input_dir,
        label_dir=label_dir,
        phase="cln",
        batch_size=batch_size,
        total=args.total_num,
        size=size,
    )

    # create attack
    attack = get_attack(
        attack=args.attack_method, arch=arch, model=model, loss_fn=F.cross_entropy, args=args
    )

    # generate adversarial example
    model.eval()
    if args.gamma < 1.0:  # use Skip Gradient Method (SGM)
        attack.register_hook()
    for inputs, labels, indexs in tqdm(data_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        with torch.no_grad():
            _, preds = torch.max(model(inputs), dim=1)

        inputs_adv = attack.perturb(inputs, preds)
    
        # save adversarial example
        save_image(
            images=inputs_adv.detach().cpu().numpy(),
            indexs=indexs,
            img_list=img_list,
            output_dir=args.output_dir,
        )




def main():
    _args = get_args()

    source_model_config = configs.get_source_model_config(_args.dataset)
    target_model_config = configs.get_target_model_config(_args.dataset)
    if _args.source_model == "":
        _args.source_model = source_model_config.keys()
    if _args.target_model == "":
        _args.target_model = target_model_config.keys()
    assert set(_args.source_model).issubset(set(source_model_config.keys()))
    assert set(_args.target_model).issubset(set(target_model_config.keys()))

    white_arguments = ['attack_method', 'total_num', 'target', 'eps', 'nb_iter', 'eps_iter']
    black_arguments = ['input_dir', 'output_dir', 'source_model', 'target_model', 'print_freq', 'not_valid', 'inputfolder', 'dataset']
    log_name = [_args.dataset]
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

    # generate adversarial examples
    logger.info(f"Generate adversarial examples with {_args.attack_method}")
    for i, source_model_name in enumerate(_args.source_model):
        logger.info(f"Attacking {source_model_name}...")

        # make output dir
        output_dir = os.path.join(output_root_dir, source_model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _args.output_dir = output_dir

        # load config
        model_config = source_model_config[source_model_name]
        if _args.attack_method.endswith("_fgsm"):
            attack_method_config = getattr(configs, "fgsm_base")
        elif _args.attack_method.startswith("sgm"):
            attack_method_config = getattr(configs, "sgm_base")
        else:
            raise NotImplementedError(f"No such attack method: {_args.attack_method}")
        args = vars(_args)
        args = {
            **model_config,
            **attack_method_config, 
            **{k: args[k] for k in args if (args[k] is not None and '_model' not in k)}
        } 

        args = Parameters(args)
        args.eps /= 255.0
        args.eps_iter /= 255.0
        logger.info(args)

        # begin attack
        logger.info(
            f"[{i+1} / {len(_args.source_model)}] source model: {source_model_name}"
        )
        attack_source_model(source_model_name, args)
        logger.info(f"Attack finished.")

        # validate
        acc_list = []
        if not args.not_valid:
            for target_model_name in _args.target_model:
                if target_model_name == "robust_models":
                    correct_cnt, model_name = evaluate_with_robust_model(args.output_dir)
                    for i in range(len(model_name)):
                        suc_rate = correct_cnt[i] * 100.0 / args.total_num
                        acc_list.append(suc_rate)
                        logger.info(f"{model_name[i]}: {suc_rate:.2f}%")

                else:
                    logger.info(f"Transfer to {target_model_name}..")
                    cln_acc, adv_acc, suc_rate = evaluate_with_natural_model(target_model_name, args.dataset, args.input_dir, args.output_dir, args.total_num)
                    acc_list.append(suc_rate)
                    # logger.info(f"clean acc:    {cln_acc:.2f}%")
                    logger.info(f"success rate: {suc_rate:.2f}%")
                    logger.info(f"Transfer done.")

        torch.cuda.empty_cache()

        logger.info("\t".join([str(round(v, 2)) for v in acc_list]))
        logger.info(round(
            (sum(acc_list) - acc_list[configs.target_model_names.index(source_model_name)]) / (len(configs.target_model_names) - 1), 2)
        )


if __name__ == "__main__":
    main()
