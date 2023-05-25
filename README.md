# Autofocusing of Multispectral Microscopic Imagery Using Deep Learning
## Introduction
This is the repo of Autofocusing of Multispectral Microscopic Imagery Using Deep Learning. The frameworks of Pix2Pix and cycleGAN were modified from **[ PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)** by [eriklindernoren](https://github.com/eriklindernoren). 
The codes were developed by me (E-Mail: [ge35sog@mytum.de](mailto:ge35sog@mytum.de)), under the supervision of Mr. Xingchen Dong (E-Mail: [xingchen.dong@tum.de](mailto:xingchen.dong@tum.de)) and Mr. Hongwei Li (Email: [hongwei.li@tum.de](mailto:hongwei.li@tum.de)).

![pix2pix](https://github.com/Jo0920/Autofocusing_of_Multispectral_Microscopic_Imagery_Using_Deep_Learning/blob/main/imgs/pix2pix.png)
Pix2Pix

![cycleGAN](https://github.com/Jo0920/Autofocusing_of_Multispectral_Microscopic_Imagery_Using_Deep_Learning/blob/main/imgs/cyclegan.png)
cycleGAN

**Input:** N channels of defocused multi-spectral images of [N,2048,1536] size. In the training, they would be cut into [N,256,256] tensors. 
N is set to 6 by default, since 6 different wavelengths filters are used in this project (500nm,520nm, 540nm, 56nm, 580nm, 600nm). 

**Output:** multi-spectral in-focused images

**Model:** Pix2Pix (paired) and cycleGAN (unpaired) defined in `model/pix2pix.py` and `model/cycle_gan.py`.

**Loss function:** Mix of MSE and MS-SSIM

The generator is implemented based on U-Net and the discriminator is as common CNN network.
For Pix2Pix, Bayesian Neural Network version for Pix2Pix is offered in case you wish to add uncertainty and gather complexity of the model.
You can find the structures of the deterministic models in `model/unet_deterministic_models.py` and Bayesian models in `model/unet_bayesian_models.py`.

## How to train the network

First install the environment in `requirements.txt`. Note that this network needs CUDA to accelerate the training process.

Then choose the proper model and loss for your training process in `train.py`. Run this file to train the model. 

You will need to set all other parameters like learning rate, epochs in `cfg.yaml`.


## Inference 
Run `predict.py` for a deterministic model,
`predict_with_uncertainty.py` for a Bayesian model.