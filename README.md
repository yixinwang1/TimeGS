## Forecasting as Rendering: A 2D Gaussian Splatting Framework for Time Series Forecasting

The code is a PyTorch implementation of our paper "Forecasting as Rendering: A 2D Gaussian Splatting Framework for Time Series Forecasting". 


## Introduction
We propose TimeGS, a unified framework that adapts the  2D Gaussian Splatting (2DGS) rendering pipeline for time series forecasting. 

Specifically, we reshape 1D historical sequences into 2D tensors and employ multiple 2D Variation Feature Extraction (2D-VFE) blocks, which are based on UNet, to extract coupled variation features from diverse temporal views. 
These features are uniformly decoded by the Multi-Basis Gaussian Kernel Generation (MB-GKG) block to obtain the shape and intensity of Gaussian kernels, which are subsequently rasterized by the Multi-Period Chronologically Continuous Rasterization (MP-CCR) block. Finally, the Channel-Adaptive Aggregation (CCA) block fuses the forecasts via channel-adaptive weighting.

Furthermore, to address the training instability often associated with optimizing free-floating Gaussian kernels on noisy data, our Multi-Basis Gaussian Kernel Generation reformulates shape regression as a stable dictionary learning task using a fixed basis bank.

Moreover, our Multi-Period Chronologically Continuous Rasterization treats Gaussian kernels as continuous signal segments that naturally wrap around the 2D grid boundaries. This ensures that the rendered output maintains strict temporal continuity, reconciling the representational benefits of 2D structural modeling with the sequential nature of time series effectively. 

![The architecture of TimeGS](./figs/diagram.png#pic_center)


## Requirements
To install all dependencies, you can execute the following command:
```
pip install -r requirements.txt
```


## Datasets
Download the required datasets and place them under `../dataset`. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing).


## Quick Demos
We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:
```
bash scripts/long_term_forecast/ETTh1.sh
```


