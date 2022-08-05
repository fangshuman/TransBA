import os
import argparse
import random
from glob import glob
from PIL import Image
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-dir', type=str, required=True, help='ImageNet-val Dataset')
    parser.add_argument('--num-per-cls', type=int, default=1)
    parser.add_argument('--save-dir', type=str, default='data')

    args = parser.parse_args()
    return args


def main(args):
    classes = sorted(glob(args.imagenet_dir + '/*'))

    for cls in tqdm(classes):
        img_list = sorted(glob(cls + '/*.JPEG'))
        total = len(img_list)
        if args.num_per_cls <= total:
            selected_idx = random.sample(range(len(img_list)), args.num_per_cls)

            for idx in selected_idx:
                class_id, file_id = img_list[idx].split('.')[0].split('/')[-2:]
                with open(img_list[idx], 'rb') as f:
                    with Image.open(f) as img:
                        resized_img = img.resize([224, 224])
                        resized_img.save(os.path.join(args.save_dir, class_id + '_' + file_id + '.png'))
        else:
            raise NotImplementedError(f'This class has not enough images.')

    print('Finished!')
                    

if __name__ == '__main__':
    args = get_args()
    main(args)
