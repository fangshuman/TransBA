# Transferable Attack Framework
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
+ [VMI (CVPR'21)](https://arxiv.org/pdf/2103.15571) 
+ [Admix (ICCV'21)](https://arxiv.org/pdf/2102.00436.pdf)
+ [FIA (ICCV'21)](https://arxiv.org/pdf/2107.14185.pdf)
+ [EMI (arxiv'21)](http://arxiv.org/pdf/2103.10609.pdf)



## Quick Start

<!-- ### Prepare Models -->

### Data

#### ImageNet Dataset
Run `./select_images.py` to randomly sample one image per class to get images from ImageNet-val dataset,
and the selected images will be put into the path `data/`.
```
python select_images.py \
        --imagenet-dir ImageNet2012/val \
        --num-per-cls 1 \
        --save-dir data
```
If you want to use ImageNet Dataset, 
please make sure the running command is `--label-dir ./data/imagenet_class_to_idx.npy`.

#### ImageNet-compatible dataset in the NIPS 2017 adversarial competition
You can also use another common dataset.
To get this dataset please follow the instruction in [here](https://github.com/qilong-zhang/Patch-wise-iterative-attack/tree/master/dataset).
If you want to use NIPS 2017 Dataset, 
please make sure the running command is `--label-dir ./data/TrueLabel.npy`.

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


## Experiment Results

We show some experiment results on 1K randomly seletected ImageNet images,
so the reproductive results may be a little different.
And all results are under the default settings, i.e., `eps=16`, `nb_iter=10`, `step_size=1.6`.
Some attack methods may adjust the attack configs in their papers.
Here we unify all parameters.

### source model: VGG16
+ The results for existing attack methods
|            | VGG16 | RN152 | IncV3 | IncV4 | IncResV2 |
|------------|:-----:|:-----:|:-----:|:-----:|:--------:|
| I-FGSM     |  99.7 |  27.7 |  24.1 |  31.7 |   18.2   |
| MI-FGSM    |  99.7 |  51.1 |  53.0 |  57.3 |   41.3   |
| TIM        |  99.7 |  28.3 |  29.8 |  36.1 |   23.2   |
| DIM        |  99.7 |  35.0 |  32.8 |  37.3 |   23.7   |
| NI-SI-FGSM | 100.0 |  71.8 |  77.6 |  80.9 |   66.4   |
| Patch-wise |  99.9 |  45.2 |  54.3 |  51.8 |   39.8   |
| SGM        |   -   |   -   |   -   |   -   |    -     |
| VMI        |  99.9 |  68.2 |  69.6 |  72.2 |   59.1   |
| Admix      |  99.7 |  47.9 |  48.9 |  52.2 |   38.5   |
| FIA        | 100.0 |  64.5 |  57.4 |  71.3 |   49.4   |
| EMI        | 100.0 |  59.6 |  59.6 |  65.7 |   51.8   |

+ The results of combination of existing methods with DI, TI, and MI (i.e., DTM). Note that VMI and EMI have contained momentum (MI).
|             | VGG16 | RN152 | IncV3 | IncV4 | IncResV2 |
|------------ |:-----:|:-----:|:-----:|:-----:|:--------:|
| MI-DI       |  99.7 |  58.9 |  64.7 |  66.4 |   52.5   |
| MI-TI       |  99.7 |  51.8 |  57.7 |  58.1 |   45.1   |
| MI-DI-TI    |  99.7 |  62.1 |  68.3 |  68.9 |   57.1   |
| Patch + DTM | 100.0 |  53.5 |  62.8 |  60.3 |   46.0   |
| SGM + DTM   |   -   |   -   |   -   |   -   |    -     |
| VMI + DT    | 100.0 |  63.4 |  69.7 |  70.0 |   60.0   |
| Admix + DTM |  99.9 |  78.8 |  87.7 |  87.6 |   79.6   |
| FIA + DTM   | 100.0 |  70.3 |  80.4 |  82.9 |   69.0   |
| EMI + DT    | 100.0 |  74.0 |  80.0 |  82.5 |   69.1   |



### source model: ResNet-152
+ The results for existing attack methods
|            | VGG16 | RN152 | IncV3 | IncV4 | IncResV2 |
|------------|:-----:|:-----:|:-----:|:-----:|:--------:|
| I-FGSM     |  43.4 |  99.9 |  27.4 |  22.6 |   24.0   |
| MI-FGSM    |  72.0 |  99.9 |  58.3 |  50.7 |   50.3   |
| TIM        |  44.1 |  99.9 |  35.6 |  32.1 |   29.4   |
| DIM        |  57.8 | 100.0 |  47.1 |  44.4 |   41.2   |
| NI-SI-FGSM |  84.8 | 100.0 |  82.9 |  75.3 |   77.8   |
| Patch-wise |  74.1 | 100.0 |  63.6 |  55.3 |   52.1   |
| SGM        |  78.0 | 100.0 |  53.8 |  51.9 |   47.7   |
| VMI        |  83.4 | 100.0 |  78.0 |  72.3 |   73.2   |
| Admix      |  64.6 |  98.6 |  53.4 |  45.1 |   43.6   |
| FIA        |  86.3 | 100.0 |  74.2 |  71.8 |   69.6   |
| EMI        |  84.4 | 100.0 |  71.4 |  64.9 |   68.1   |

+ The results of combination of existing methods with DI, TI, and MI (i.e., DTM). Note that VMI and EMI have contained momentum (MI).
|             | VGG16 | RN152 | IncV3 | IncV4 | IncResV2 |
|------------ |:-----:|:-----:|:-----:|:-----:|:--------:|
| MI-DI       |  83.2 | 100.0 |  76.8 |  72.8 |   71.7   |
| MI-TI       |  70.6 |  99.9 |  66.0 |  60.8 |   58.9   |
| MI-DI-TI    |  82.0 | 100.0 |  79.6 |  75.3 |   76.0   |
| Patch + DTM |  85.9 | 100.0 |  78.8 |  72.8 |   71.1   |
| SGM + DTM   |  89.3 | 100.0 |  82.5 |  77.0 |   77.1   |
| VMI + DT    |  82.3 | 100.0 |  81.1 |  75.4 |   75.9   |
| Admix + DTM |  89.1 |  98.0 |  91.5 |  89.5 |   89.4   |
| FIA + DTM   |  90.8 |  99.9 |  89.7 |  87.1 |   85.5   |
| EMI + DT    |  92.1 | 100.0 |  89.2 |  86.3 |   86.0   |



### source model: InceptionV3
+ The results for existing attack methods
|            | VGG16 | RN152 | IncV3 | IncV4 | IncResV2 |
|------------|:-----:|:-----:|:-----:|:-----:|:--------:|
| I-FGSM     |  28.3 |  17.5 | 100.0 |  26.8 |   24.0   |
| MI-FGSM    |  54.3 |  38.1 | 100.0 |  50.0 |   49.0   |
| TIM        |  31.7 |  21.9 |  99.9 |  27.6 |   23.5   |
| DIM        |  38.6 |  28.2 |  99.7 |  43.6 |   40.1   |
| NI-SI-FGSM |  75.1 |  64.7 | 100.0 |  76.7 |   77.3   |
| Patch-wise |  71.8 |  48.7 | 100.0 |  55.1 |   49.7   |
| SGM        |   -   |   -   |   -   |   -   |    -     |
| VMI        |  67.6 |  59.1 | 100.0 |  73.9 |   70.5   |
| Admix      |  43.4 |  32.8 |  99.5 |  41.3 |   40.1   |
| FIA        |  75.2 |  66.5 |  96.6 |  80.0 |   77.1   |
| EMI        |  68.3 |  48.5 | 100.0 |  66.4 |   66.7   |

+ The results of combination of existing methods with DI, TI, and MI (i.e., DTM). Note that VMI and EMI have contained momentum (MI).
|             | VGG16 | RN152 | IncV3 | IncV4 | IncResV2 |
|------------ |:-----:|:-----:|:-----:|:-----:|:--------:|
| MI-DI       |  65.1 |  53.6 |  99.8 |  68.2 |   65.8   |
| MI-TI       |  62.2 |  44.3 | 100.0 |  51.3 |   47.6   |
| MI-DI-TI    |  73.9 |  59.0 |  99.8 |  69.5 |   66.6   |
| Patch + DTM |  79.9 |  62.8 | 100.0 |  70.3 |   63.9   |
| SGM + DTM   |   -   |   -   |   -   |   -   |    -     |
| VMI + DT    |  73.9 |  64.6 |  99.6 |  73.0 |   69.8   |
| Admix + DTM |  87.9 |  81.8 |  99.6 |  88.2 |   85.4   |
| FIA + DTM   |  84.9 |  75.9 |  95.8 |  80.6 |   79.4   |
| EMI + DT    |  83.6 |  72.5 | 100.0 |  83.6 |   81.1   |



### source model: InceptionV4
+ The results for existing attack methods
|            | VGG16 | RN152 | IncV3 | IncV4 | IncResV2 |
|------------|:-----:|:-----:|:-----:|:-----:|:--------:|
| I-FGSM     |  32.2 |  15.4 |  27.1 |  99.9 |   22.3   |
| MI-FGSM    |  58.9 |  36.3 |  53.4 |  99.8 |   47.3   |
| TIM        |  37.6 |  20.4 |  28.7 |  99.6 |   23.8   |
| DIM        |  42.5 |  25.4 |  42.8 |  98.1 |   37.7   |
| NI-SI-FGSM |       |       |       |       |          |
| Patch-wise |  73.7 |  44.7 |  53.9 |  99.8 |   48.7   |
| SGM        |   -   |   -   |   -   |   -   |    -     |
| VMI        |  71.8 |  56.5 |  77.0 |  99.9 |   71.5   |
| Admix      |  56.7 |  40.9 |  61.4 |  98.8 |   52.3   |
| FIA        |   -   |   -   |   -   |   -   |    -     |
| EMI        |  71.1 |  44.8 |  67.0 | 100.0 |   62.6   |

+ The results of combination of existing methods with DI, TI, and MI (i.e., DTM). Note that VMI and EMI have contained momentum (MI).
|             | VGG16 | RN152 | IncV3 | IncV4 | IncResV2 |
|------------ |:-----:|:-----:|:-----:|:-----:|:--------:|
| MI-DI       |  68.3 |  49.8 |  70.5 |  97.9 |   65.3   |
| MI-TI       |  66.1 |  41.6 |  56.2 |  99.6 |   47.4   |
| MI-DI-TI    |  75.3 |  55.7 |  72.7 |  97.7 |   64.9   |
| Patch + DTM |  82.8 |  60.7 |  73.7 |  98.4 |   64.8   |
| SGM + DTM   |   -   |   -   |   -   |   -   |    -     |
| VMI + DT    |  76.3 |  61.0 |  75.3 |  98.5 |   69.2   |
| Admix + DTM |  85.0 |  79.5 |  87.6 |  96.8 |   84.7   |
| FIA + DTM   |   -   |   -   |   -   |   -   |    -     |
| EMI + DT    |  87.1 |  69.1 |  85.0 | 100.0 |   80.3   |



### source model: InceptionResNetV2
+ The results for existing attack methods
|            | VGG16 | RN152 | IncV3 | IncV4 | IncResV2 |
|------------|:-----:|:-----:|:-----:|:-----:|:--------:|
| I-FGSM     |  30.0 |  17.7 |  29.5 |  26.8 |   97.9   |
| MI-FGSM    |  57.9 |  39.9 |  58.2 |  52.0 |   97.9   |
| TIM        |  37.0 |  24.8 |  34.6 |  30.4 |   97.5   |
| DIM        |  39.6 |  29.5 |  46.2 |  45.2 |   94.0   |
| NI-SI-FGSM |       |       |       |       |          |
| Patch-wise |  74.6 |  51.2 |  60.6 |  55.6 |   98.8   |
| SGM        |   -   |   -   |   -   |   -   |    -     |
| VMI        |  69.6 |  63.1 |  75.8 |  73.9 |   98.1   |
| Admix      |  52.5 |  46.0 |  63.5 |  55.4 |   96.7   |
| FIA        |  67.9 |  61.0 |  70.7 |  68.3 |   85.1   |
| EMI        |  70.0 |  51.0 |  69.9 |  65.9 |   99.5   |

+ The results of combination of existing methods with DI, TI, and MI (i.e., DTM). Note that VMI and EMI have contained momentum (MI).
|             | VGG16 | RN152 | IncV3 | IncV4 | IncResV2 |
|------------ |:-----:|:-----:|:-----:|:-----:|:--------:|
| MI-DI       |  63.7 |  53.8 |  68.2 |  64.2 |   94.5   |
| MI-TI       |  67.2 |  50.4 |  61.8 |  56.8 |   97.5   |
| MI-DI-TI    |  72.7 |  60.7 |  73.2 |  68.4 |   94.3   |
| Patch + DTM |  83.0 |  68.9 |  77.1 |  72.4 |   95.7   |
| SGM + DTM   |   -   |   -   |   -   |   -   |    -     |
| VMI + DT    |  73.0 |  65.9 |  73.8 |  71.8 |   95.0   |
| Admix + DTM |  84.8 |  83.6 |  88.4 |  86.6 |   94.2   |
| FIA + DTM   |  77.9 |  68.5 |  72.2 |  69.3 |   82.4   |
| EMI + DT    |  86.4 |  77.2 |  85.5 |  83.6 |   99.2   |

