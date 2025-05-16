# Grayscale-Assisted RGB Image Conversion from Near-Infrared Images
We provide the Pytorch implementation of Grayscale-Assisted RGB Image Conversion from Near-Infrared Images.

## Abstract
Recent methods aim to recover the corresponding RGB image directly from the NIR image using Convolutional Neural Networks. 
However, these methods struggle with accurately recovering both luminance and chrominance information and the inherent deficiencies in NIR image details. 
In this paper, we propose grayscale-assisted RGB image restoration from NIR images to recover luminance and chrominance information in two stages. 
We address the complex NIR-to-RGB conversion challenge by decoupling it into two separate stages. First, it converts NIR-to-grayscale images, focusing on luminance learning. Then, it transforms grayscale-to-RGB images, concentrating on chrominance information. 
In addition, we incorporate frequency domain learning to shift the image processing from the spatial domain to the frequency domain, facilitating the restoration of the detailed textures often lost in NIR images. 
Empirical evaluations of our grayscale-assisted framework against existing state-of-the-art methods demonstrate its superior performance and yield more visually appealing results.

## Highlights
+ We present a novel framework designed to tackle the NIR2RGB task. It simplifies the process by decoupling it into two phases: NIR2GRAY for luminance recovery and GRAY2RGB for chrominance restoration.

<img width="350" alt="image" src="https://github.com/Yiiclass/RING/assets/69071622/832d9b00-11a0-4fe7-a117-e14c5356f38e">

+ To effectively counteract the detail deficiencies typical in NIR images, we integrate the FDL module. This technique enhances detail and edge clarity by shifting image features from the spatial to the frequency domain.

<img width="709" alt="image" src="https://github.com/Yiiclass/RING/assets/69071622/373e5fbb-0ef7-4677-a8a0-0039bbd65b12">

+ Our method effectively recovers the corresponding RGB images from NIR images. Quantitative and qualitative experiments indicate that our method outperforms state-of-the-art methods and yields more visually appealing results.


## Dataset
The datasets used for evaluation are [ICVL](https://icvl.cs.bgu.ac.il/hyperspectral/), [TokyoTech](http://www.ok.sc.e.titech.ac.jp/res/MSI/MSIdata31.html) and [IDH](https://github.com/cccyz/NIR2RGB).

```
dataset
+-- 'train'
|   +-- train_ICVL_1.png
|   +-- train_ICVL_2.png
|   +-- train_ICVL_3.png
|   +-- train_ICVL_4.png
|   +-- train_ICVL_5.png
|   +-- train_ICVL_6.png
|   +-- ...
+-- 'test'
|   +-- test_ICVL_1.png
|   +-- test_ICVL_2.png
|   +-- test_ICVL_3.png
|   +-- test_ICVL_4.png
|   +-- test_ICVL_5.png
|   +-- test_ICVL_6.png
|   +-- ...
```

For example:

![ICVL_1](https://github.com/user-attachments/assets/a15a4af8-0cc1-486f-a6c7-2a938e14e726)


## Usage
+ Create conda environment and download our repository

```
conda create -n eventhdr python=3.7
conda activate RING
git clone https://github.com/Yiiclass/RING
cd RING
```

### Requirments
1. Ubuntu 16.04
2. CUDA 9.1
3. pytorch 1.7.1

### Training
```
CUDA_VISIBLE_DEVICES=0  python train.py \
--dataroot Datasets \
--name try_first \
--batch_size 2 \
--lr 0.001 \
--epochs 501 \
--ngf 50 \
--n_epochs 200 \
--n_epochs_decay 300
```



### Testing
```
CUDA_VISIBLE_DEVICES=0  python test.py \
--dataroot Datasets \
--name try_first \
--test_epoch best 
```


## Results

### Quantitative comparison on the ICVL dataset


| Method                                | PSNR ↑     | SSIM ↑    | RMSE ↓   | Delta-E ↓ | FLOPs (G) ↓ | Parameters (M) ↓ | Inference time (s) ↓ |
|---------------------------------------|------------|-----------|----------|-----------|-------------|------------------|-----------------------|
| Retinex-Net                           | 8.99       | 0.23      | 10.35    | 13.06     | 587.47      | 0.55             | 0.142                 |
| LIME                                  | 12.26      | 0.46      | 10.55    | 11.00     | -           | -                | 2.997                 |
| DUA                                   | 11.86      | 0.39      | 10.52    | 11.95     | -           | -                | 6.042                 |
| MBLLE                                 | 12.26      | 0.46      | 10.55    | 11.00     | 301.12      | **0.45**         | 0.640                 |
| Ideepcolor                            | 12.52      | 0.52      | 10.55    | 10.49     | 146.32      | 34.19            | 0.081                 |
| CIC                                   | 12.96      | 0.52      | 10.54    | 10.54     | 48.29       | 32.24            | 0.070                 |
| Jointcolor                            | 16.95      | 0.55      | 9.85     | 14.97     | **8.60**    | 42.3             | 0.032                 |
| CT2                                   | 18.42      | 0.71      | 9.67     | 12.73     | 1784.47     | 462.98           | 0.175                 |
| CUT                                   | 20.50      | 0.62      | 8.80     | 10.03     | 70.64       | 14.70            | **0.011**             |
| UNIT                                  | 21.58      | 0.70      | 9.05     | 11.14     | 142.29      | 38.82            | 0.355                 |
| CycleGAN                              | 22.31      | 0.76      | 8.42     | 8.18      | 28.29       | 120.12           | 0.013                 |
| BBDM                                  | 24.00      | 0.84      | 8.22     | 7.92      | 1981.09     | 237.09           | 32.262                |
| NIRcolor                              | 16.37      | 0.69      | 9.90     | 26.62     | 206.33      | 53.62            | 0.007                 |
| RVM                                   | 24.21      | 0.80      | 8.01     | 7.33      | 21.36       | 57.20            | 0.037                 |
| TLM                                   | 24.53      | 0.79      | 7.84     | 7.28      | 21.36       | 57.20            | 0.031                 |
| Ours                                  | **25.26**  | **0.85**  | **7.83** | **6.89**  | 19.84       | 38.57            | 0.032                 |



## Citation

If you find this work useful for your research, please cite:

```
@article{gao2025grayscale,
  title={Grayscale-Assisted RGB Image Conversion from Near-Infrared Images}, 
  author={Gao, Yunyi and Liu, Qiankun and Gu, Lin and Fu, Ying},
  journal={Tsinghua Science and Technology},
  volume={30},
  number={5},
  pages={2215--2226},
  year={2025},
  publisher={TUP},
  doi={10.26599/TST.2024.9010115}
}

```

## contact
If you have any problems, please feel free to contact me at yiiclass@qq.com
