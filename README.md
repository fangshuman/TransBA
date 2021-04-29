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
+ PGD
+ [TIM](https://arxiv.org/pdf/1904.02884)
+ [DIM](https://arxiv.org/pdf/1803.06978)
+ [MI-FGSM](https://arxiv.org/pdf/1710.06081)
+ [NI-SI-FGSM](https://arxiv.org/pdf/1908.06281)  not yet
+ [ILA](https://arxiv.org/pdf/1907.10823)
+ [SGM](https://arxiv.org/pdf/2002.05990)
+ [VI](https://arxiv.org/pdf/2103.15571)  not yet



## Quick Start

### Prepare the Data and Models

#### Data

To run code with 1k images from ImageNet you should download [1K Images](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) and extrace images to the path `dataset_1000/` and with 5k images should download [5K Images](https://drive.google.com/file/d/1RqDUGs7olVGYqSV_sIlqZRRhB9Mw48vM/view?usp=sharing) and place images to `dataset_5000/` respectively. Make sure the file name format of image is like n01440764_ILSVRC2012_val_00007197.png

#### Pretrainedmodels

All pretrained models can be found online: [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)


### Run the Code

Generate adversairal examples and save them into path `./output/`. The running log will be saved in `./output_log`. This code is to generate adversarial examples using I-FGSM on all source models separately and validate the accuracy on all target models.

```
python main.py --attack-method i_fgsm
```

To choose some of source models or some of target models, 
```
python main.py --attack-method i_fgsm --source-model resnet50 densenet121 --target-model resnet152 densenet201 inceptionv3 inceptionv4 inceptionresnetv2
```



## Results

### Result from 1k Images

For all methods we set `eps=16/255, nb_iter=10, eps_iter=1.6/255`.

| Source Model | Method  | VGG16_BN | RN-50 | RN-101 | RN-152 | DN-121 | DN-169 | DN-201 | Inc-v3 | Inc-v4 | IncRes-v2 |
| :----------- | :-----: | :------: | :---: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :-------: |
| VGG16_BN     | I-FGSM  |   0.0    | 66.0  |  76.6  |  78.5  |  65.9  |  70.0  |  75.8  |  82.2  |  82.1  |   86.6    |
|              |   TIM   |   0.0    | 57.2  |  66.3  |  67.6  |  53.0  |  58.9  |  61.7  |  67.2  |  65.8  |   69.8    |
|              |   DIM   |   0.1    | 53.1  |  65.4  |  69.1  |  53.5  |  59.8  |  65.0  |  75.9  |  71.5  |   78.8    |
|              | MI-FGSM |   0.1    | 42.2  |  55.5  |  60.2  |  39.7  |  46.5  |  50.5  |  62.8  |  59.2  |   68.1    |
| RN-50        | I-FGSM  |   39.7   |  0.0  |  26.6  |  34.4  |  38.2  |  43.0  |  44.0  |  74.6  |  75.3  |   77.7    |
|              |   TIM   |   41.1   |  0.0  |  31.9  |  34.8  |  34.2  |  37.7  |  39.8  |  57.3  |  63.9  |   62.3    |
|              |   DIM   |   26.9   |  0.0  |  17.5  |  20.7  |  24.5  |  27.0  |  29.2  |  60.3  |  63.4  |   65.6    |
|              | MI-FGSM |   19.9   |  0.0  |  11.0  |  14.9  |  15.0  |  16.7  |  18.2  |  47.0  |  51.1  |   54.4    |
| DN-121       | I-FGSM  |   34.1   | 33.6  |  42.8  |  45.0  |  0.0   |  18.1  |  22.6  |  68.7  |  70.4  |   74.4    |
|              |   TIM   |   41.8   | 35.9  |  44.4  |  45.6  |  0.0   |  25.7  |  28.3  |  52.8  |  58.5  |   61.1    |
|              |   DIM   |   22.0   | 19.6  |  27.9  |  30.9  |  0.0   |  11.9  |  14.4  |  50.7  |  54.1  |   60.2    |
|              | MI-FGSM |   17.4   | 16.0  |  23.3  |  25.4  |  0.1   |  6.8   |  9.0   |  41.6  |  44.2  |   52.7    |
| Inc-v3       | I-FGSM  |   65.5   | 76.9  |  79.8  |  81.2  |  75.2  |  77.2  |  79.4  |  0.1   |  71.6  |   73.4    |
|              |   TIM   |   60.1   | 68.4  |  72.5  |  73.0  |  67.6  |  69.0  |  69.9  |  0.2   |  69.8  |   69.7    |
|              |   DIM   |   53.1   | 66.1  |  70.1  |  71.5  |  64.5  |  65.7  |  70.2  |  0.5   |  55.9  |   61.1    |
|              | MI-FGSM |   45.4   | 57.2  |  60.5  |  63.9  |  55.3  |  56.7  |  57.1  |  0.1   |  51.9  |   53.1    |
| Inc-v4       | I-FGSM  |   60.8   | 79.2  |  82.0  |  82.5  |  76.0  |  77.4  |  80.2  |  69.9  |  0.0   |   73.3    |
|              |   TIM   |   54.6   | 69.3  |  73.1  |  72.4  |  68.6  |  68.3  |  71.8  |  63.8  |  1.4   |   65.5    |
|              |   DIM   |   47.2   | 67.6  |  71.6  |  71.9  |  66.0  |  65.4  |  69.9  |  53.2  |  1.9   |   58.5    |
|              | MI-FGSM |   39.8   | 58.5  |  62.3  |  62.7  |  53.0  |  58.3  |  59.2  |  48.0  |  0.1   |   50.5    |
| IncRes-v2    | I-FGSM  |   64.1   | 76.3  |  80.5  |  80.6  |  75.6  |  77.2  |  78.3  |  70.2  |  70.1  |    2.0    |
|              |   TIM   |   56.4   | 68.5  |  71.6  |  71.1  |  66.5  |  68.0  |  69.3  |  62.3  |  65.3  |    4.2    |
|              |   DIM   |   53.4   | 64.1  |  68.9  |  67.6  |  63.0  |  65.6  |  67.1  |  52.1  |  53.2  |    6.3    |
|              | MI-FGSM |   43.2   | 53.3  |  60.0  |  60.2  |  52.3  |  56.2  |  56.2  |  44.5  |  49.1  |    2.6    |