# Transfer Attack Framework
This repository contains code to reproduce results from some basic transfer attack methods.



## Requirements

+ python >= 3.6.5
+ torch >= 1.7.0
+ torchvision >= 0.8.2
+ pretrainedmodels >= 0.7.4
+ numpy >= 1.19.5
+ scipy > 1.5.4



## Supported Methods

+ I-FGSM 
+ [TIM](https://arxiv.org/pdf/1904.02884)
+ [DIM](https://arxiv.org/pdf/1803.06978)
+ [MI-FGSM](https://arxiv.org/pdf/1710.06081)
+ [NI-SI-FGSM](https://arxiv.org/pdf/1908.06281) 
+ [VMI-FGSM](https://arxiv.org/pdf/2103.15571) 
+ [Patch-wise](https://arxiv.org/pdf/2007.06765)
+ [SGM](https://arxiv.org/pdf/2002.05990)
+ [Admix](https://arxiv.org/pdf/2102.00436.pdf)
+ [FIA](https://arxiv.org/pdf/2107.14185.pdf)
+ [ILA](https://arxiv.org/pdf/1907.10823)




## Quick Start

### Prepare Models

<!-- #### Data

To run code with 1k images from ImageNet you should download [1K Images](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) and extrace images to the path `dataset_1000/` while with 5k images, download [5K Images](https://drive.google.com/file/d/1RqDUGs7olVGYqSV_sIlqZRRhB9Mw48vM/view?usp=sharing) and place images into `dataset_5000/` respectively. Make sure the file name format of image is like n01440764_ILSVRC2012_val_00007197.png -->

<!-- #### Pretrainedmodels -->

All pretrained models can be found online: [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)


### Run the Code

Generate adversairal examples and save them into path `./output/`. The running log will be saved in `./output_log`. This code is to generate adversarial examples using I-FGSM on all source models separately and validate the attack success rate on all target models.

If the prediction of adversarial example is different from clean one, it will be considered as attacking successfully.
And the 'y' used during attack process are the predictions of clean imgaes, so the true labels of images are not necessary.


#### About attack method

```
python3 main.py \
        --attack-method i_fgsm \
        --source-model xxx xxx xxx (optional) \
        --target-model xxx xxx xxx xxx xxx (optinal)
```

#### I-FGSM based

+ I-FGSM:  `--attack-method i_fgsm`
+ MI-FGSM: `--attack-method mi_fgsm`
+ DI-FGSM: `--attack-method di_fgsm`
+ TI-FGSM: `--attack-method ti_fgsm`
+ combine: `--attack-method mi_di_ti_fgsm`
  
#### VMI based

+ VMI-FGSM: `--attack-method vi_mi_fgsm`
+ combine:  `--attack-method vi_mi_xxx_fgsm`

#### Patch-wise based

+ Patch-wise: `--attack-method pi_fgsm`
+ combine   : `--attack-method pi_xxx_fgsm`

#### SGM based

+ SGM: `--attack-method sgm`
+ combine: `--attack-method sgm_xxx`

#### Admix based

+ Admix: `--attack-method admix`
+ combine: `--attack-method admix_xxx`
  
#### FIA based

+ FIA: `--attack-method fia`
+ combine: `--attack-method fia_xxx`

