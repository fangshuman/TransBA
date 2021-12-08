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

from utils import Parameters
from models import make_model
from models import get_model_config
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
    parser.add_argument("--gpu-id", type=str, default="3")
    parser.add_argument(
        "--dataset", type=str, default="ImageNet", choices=["ImageNet", "CIFAR10"]
    )
    parser.add_argument("--input-dir", type=str, default="dataset_1000")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--attack-method", type=str, default="i_fgsm")
    parser.add_argument("--source-model", nargs="+", default="")
    parser.add_argument("--target-model", nargs="+", default="")
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
        "--folder-with-label",
        help="input dataset in folder dataset",
        action="store_true",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    return args


def attack_source_model(arch, args):
    # create model
    model = make_model(arch=arch, dataset=args.dataset)
    size = model.input_size[1]
    model = model.cuda()

    # create dataloader
    batch_size = args.batch_size
    img_list, data_loader = make_loader(
        image_dir=args.input_dir,
        label_dir=args.folder_with_label,
        phase="cln",
        batch_size=batch_size,
        total=args.total_num,
        size=size,
    )

    # create attack
    attack = get_attack(
        attack_method=args.attack_method,
        model=model,
        loss_fn=F.cross_entropy,
        args=args,
    )

    # generate adversarial example
    model.eval()
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
    del model


def main():
    global_args = get_args()

    source_model_config = get_model_config(global_args.dataset)
    target_model_config = get_model_config(global_args.dataset, is_source=False)
    if global_args.source_model == "":
        global_args.source_model = list(source_model_config.keys())
    if global_args.target_model == "":
        global_args.target_model = list(target_model_config.keys())
    assert set(global_args.source_model).issubset(set(source_model_config.keys()))
    assert set(global_args.target_model).issubset(set(target_model_config.keys()))

    white_arguments = [
        "attack_method",
        "total_num",
        "target",
        "eps",
        "nb_iter",
        "eps_iter",
    ]
    black_arguments = [
        "input_dir",
        "output_dir",
        "source_model",
        "target_model",
        "print_freq",
        "not_valid",
        "folder_with_label",
        "dataset",
    ]
    log_name = [global_args.dataset]
    for a in white_arguments:
        v = getattr(global_args, a)
        if v is not None:
            log_name.append(f"{a}_{v}")
    for k, v in vars(global_args).items():
        if k in white_arguments or k in black_arguments:
            continue
        if v is not None:
            log_name.append(f"{k}_{v}")

    log_name = "-".join(log_name)

    output_root_dir = os.path.join(global_args.output_dir, log_name)
    logger_path = os.path.join("output_log", log_name + ".log")

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
    logger.info(f"Generate adversarial examples with {global_args.attack_method}")
    for i, source_model_name in enumerate(global_args.source_model):
        logger.info(f"Attacking {source_model_name}...")

        # make output dir
        output_dir = os.path.join(output_root_dir, source_model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        global_args.output_dir = output_dir

        # load config
        model_config = source_model_config[source_model_name]
        attack_method_config = get_attack(global_args.attack_method).config
        args = vars(global_args)
        args = {
            **model_config,
            **attack_method_config,
            **{k: args[k] for k in args if (args[k] is not None and "_model" not in k)},
        }

        args = Parameters(args)
        args.eps /= 255.0
        args.eps_iter /= 255.0
        logger.info(args)

        # begin attack
        logger.info(
            f"[{i+1} / {len(global_args.source_model)}] source model: {source_model_name}"
        )
        attack_source_model(source_model_name, args)
        logger.info(f"Attack finished.")

        # validate
        acc_list = []
        if not args.not_valid:
            for target_model_name in global_args.target_model:
                if target_model_name == "robust_models":
                    correct_cnt, model_name = evaluate_with_robust_model(
                        args.input_dir,
                        args.output_dir
                    )
                    for i in range(len(model_name)):
                        suc_rate = correct_cnt[i] * 100.0 / args.total_num
                        acc_list.append(suc_rate)
                        logger.info(f"{model_name[i]}: {suc_rate:.2f}%")

                else:
                    logger.info(f"Transfer to {target_model_name}..")
                    cln_acc, adv_acc, suc_rate = evaluate_with_natural_model(
                        target_model_name,
                        args.dataset,
                        args.input_dir,
                        args.output_dir,
                        args.total_num,
                    )
                    acc_list.append(suc_rate)
                    # logger.info(f"clean acc:    {cln_acc:.2f}%")
                    logger.info(f"Success rate: {suc_rate:.2f}%")
                    logger.info(f"Transfer done.")

            torch.cuda.empty_cache()

            logger.info("\t".join([str(round(v, 2)) for v in acc_list]))
            result_source_in_target = (
                acc_list[global_args.target_model.index(source_model_name)]
                if source_model_name in global_args.target_model
                else 0
            )
            logger.info(
                round(
                    (sum(acc_list) - result_source_in_target)
                    / (len(global_args.target_model) - int(result_source_in_target > 0)),
                    2,
                )
            )


if __name__ == "__main__":
    main()
