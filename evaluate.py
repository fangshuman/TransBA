import os
import argparse
import logging
import ipdb

import torch
import torch.nn as nn

import configs
from models import make_model
from dataset import make_loader
from eval_robust_models import evaluate_with_robust_model


def evaluate_with_natural_model(arch, input_dir, total_num):
    model = make_model(arch=arch)
    size = model.input_size[1]
    model = model.cuda()
    model.eval()
    total = 0
    count = 0

    _, data_loader = make_loader(
        image_dir=input_dir,
        label_dir="imagenet_class_to_idx.npy",
        phase="adv",
        batch_size=configs.target_model_batch_size[arch],
        total=total_num,
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str)
    parser.add_argument('--target-model', nargs="+", default=configs.target_model_names)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--total-num", type=int, default=1000)
    args = parser.parse_args()
    
    assert set(args.target_model).issubset(set(configs.target_model_names))

    logger_path = os.path.join("output_log", "valid.log")

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
    logger.info(args)

    acc_list = []
    for target_model_name in args.target_model:
        if target_model_name == "robust_models":
            correct_cnt, model_name = evaluate_with_robust_model(args.input_dir)
            for i in range(len(model_name)):
                acc = correct_cnt[i] * 100.0 / args.total_num
                acc_list.append(acc)
                logger.info(f"{model_name[i]}: {acc:.2f}%")
        
        else:
            logger.info(f"Transfer to {target_model_name}..")
            acc = evaluate_with_natural_model(target_model_name, args.input_dir, args.total_num)
            acc_list.append(acc)
            logger.info(f"acc: {acc:.2f}%")
            logger.info(f"Transfer done.")

        torch.cuda.empty_cache()
    
    logger.info("\t".join([str(round(v, 2)) for v in acc_list]))
