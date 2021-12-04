import os
import argparse
import logging

import torch

from models import make_model
from models import get_model_config
from dataset import make_loader
# from eval_robust_models import evaluate_with_robust_model


def evaluate_with_natural_model(arch, dataset, cln_dir, adv_dir, total_num):
    target_model_config = get_model_config(dataset, is_source=False)

    model = make_model(arch=arch, dataset=dataset)
    size = model.input_size[1]
    model = model.cuda()

    label_dir = "TrueLabel.npy" if dataset == "ImageNet" else "cifar10_class_to_idx.npy"

    _, cln_data_loader = make_loader(
        image_dir=cln_dir,
        label_dir=label_dir,
        phase="cln",
        batch_size=target_model_config[arch],
        total=total_num,
        size=size,
    )

    _, adv_data_loader = make_loader(
        image_dir=adv_dir,
        label_dir=label_dir,
        phase="adv",
        batch_size=target_model_config[arch],
        total=total_num,
        size=size,
    )

    model.eval()
    total = 0
    cln_count = 0
    adv_count = 0
    success = 0
    with torch.no_grad():
        for (cln_x, cln_y, _), (adv_x, adv_y, _) in zip(cln_data_loader, adv_data_loader):
            cln_x = cln_x.cuda()
            adv_x = adv_x.cuda()

            _, cln_preds = torch.max(model(cln_x), dim=1)
            _, adv_preds = torch.max(model(adv_x), dim=1)

            total += cln_x.size(0)
            cln_count += (cln_preds.detach().cpu() == cln_y).sum().item()
            adv_count += (adv_preds.detach().cpu() == cln_y).sum().item()
            success += (cln_preds != adv_preds).sum().item()

    cln_acc = cln_count * 100.0 / total
    adv_acc = adv_count * 100.0 / total
    return cln_acc, adv_acc, success * 100.0 / total




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, default="3")
    parser.add_argument("--adver-dir", type=str)
    parser.add_argument("--dataset", type=str, default="ImageNet")
    parser.add_argument("--clean-dir", type=str, default="../dataset/NIPS_dataset")
    parser.add_argument('--target-model', nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--total-num", type=int, default=1000)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    target_model_names = set(get_model_config(args.dataset, is_source=False).keys())
    if len(args.target_model) == 0:
        args.target_model = target_model_names
    assert set(args.target_model).issubset(target_model_names)

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
            correct_cnt, model_name = evaluate_with_robust_model(args.adver_dir)
            for i in range(len(model_name)):
                acc = correct_cnt[i] * 100.0 / args.total_num
                acc_list.append(acc)
                logger.info(f"{model_name[i]}: {acc:.2f}%")

        else:
            logger.info(f"Transfer to {target_model_name}..")
            _, _, suc_rate = evaluate_with_natural_model(target_model_name, args.dataset, args.clean_dir, args.adver_dir, args.total_num)
            acc_list.append(suc_rate)
            logger.info(f"Success rate: {suc_rate:.2f}%")
            logger.info(f"Transfer done.")

        torch.cuda.empty_cache()

    logger.info("\t".join([str(round(v, 2)) for v in acc_list]))
