# TransBA: Transfer-based Black Attack Framework
This repository contains code to reproduce results from some basic transferable attack methods.


<!-- 
## Requirements

+ python >= 3.6.5
+ torch >= 1.7.0
+ torchvision >= 0.8.2
+ pretrainedmodels >= 0.7.4
+ numpy >= 1.19.5
+ scipy > 1.5.4
 -->


## Supported Methods

+ I-FGSM 
+ [MI-FGSM (CVPR'18)](https://arxiv.org/pdf/1710.06081)
+ [TIM (CVPR'19)](https://arxiv.org/pdf/1904.02884)
+ [DIM (CVPR'19)](https://arxiv.org/pdf/1803.06978)
+ [ILA (ICCV'19)](https://arxiv.org/pdf/1907.10823)
+ [NI-SI-FGSM (ICLR'20)](https://arxiv.org/pdf/1908.06281) 
+ [Patch-wise (ECCV'20)](https://arxiv.org/pdf/2007.06765)
+ [SGM (ICLR'20)](https://arxiv.org/pdf/2002.05990)
+ [VT (CVPR'21)](https://arxiv.org/pdf/2103.15571) 
+ [Admix (ICCV'21)](https://arxiv.org/pdf/2102.00436.pdf)
+ [FIA (ICCV'21)](https://arxiv.org/pdf/2107.14185.pdf)
+ [EMI (arxiv'21)](http://arxiv.org/pdf/2103.10609.pdf)



## Quick Start

<!-- ### Prepare Models -->

### Data
Run `./select_images.py` to randomly sample one image per class to get images from ImageNet-val dataset,
and the selected images will be put into the path `data/`.
```
python select_images.py \
        --imagenet-dir ImageNet2012/val \
        --num-per-cls 1 \
        --save-dir data
```

### Pretrained Models
All pretrained models can be found online: [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch).


### Run the Code
Generate adversairal examples and save them into path `./output/`, and the running log will be saved in `./output_log`. 
This code is to generate adversarial examples on all source models one by one,
and validate the attack success rate on all target models.

It will be considered as attacking successfully,
if the prediction of adversarial example is different from the clean one.
<!-- And the "y" used during attack process are the predictions of clean images, 
so the true labels of images are not necessary. -->


#### Attack
If you want test the effectiveness of I-FGSM with `ResNet-50`, `DenseNet-121` as source models and `Inception-v3`, `Inception-v4`, and `Inception-ResNet-v2` as target models, 
run as follows.
```
python3 main.py \
        --attack-method i_fgsm \
        --source-model resnet50 densenet121 \
        --target-model inceptionv3 inceptionv4 inceptionresnetv2
```

#### Combine Attack Methods
This code supports combine existing methods with basic attack methods, 
including MI-FGSM, TIM, and DIM.
For example,
run as follows to evaluate the effectiveness of MI+DI+TI.
```
python3 main.py \
        --attack-method mi_di_ti_fgsm
```

#### Evaluate
To evaluate with naturally trained models as target model:
```
python3 evaluate_NT_trained.py \
        --adver-dir path_saved_adversarial_examples
```

To evaluate with adversarially trained models as target model:
```
python3 evaluate_AT_trained.py \
        --adver-dir path_saved_adversarial_examples
```
