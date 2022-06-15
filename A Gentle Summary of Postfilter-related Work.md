# A Gentle Summary of Postfilter-related Work

# NN-based Postfilter (Decoder)
## [Inplace Gated Convolutional Recurrent Neural Network for Dual-Channel Speech Enhancement](https://www.isca-speech.org/archive/pdfs/interspeech_2021/liu21f_interspeech.pdf), Interspeech 2021

## [End-to-End Post-Filter for Speech Separation With Deep Attention Fusion Features](https://ieeexplore.ieee.org/abstract/document/9043689), TASLP 2020

## [$Y^2$-Net FCRN for Acoustic Echo and Noise Suppression](https://arxiv.org/abs/2103.17189), Interspeech 2021

## [Bandwidth-Scalable Fully Mask-Based Deep FCRN Acoustic Echo Cancellation and Postfiltering](https://arxiv.org/abs/2205.04276)
Follow $Y^2$-Net 的工作

## [Dual-Path Filter Network: Speaker-Aware Modeling for Speech Separation](https://www.isca-speech.org/archive/pdfs/interspeech_2021/wang21x_interspeech.pdf), Interspeech 2021


# Conventional
## [Nonlinear Spatial Filtering in Multichannel Speech Enhancement](https://arxiv.org/abs/2104.11033), TASLP 2021

## [A Synergistic Kalman- and Deep Postfiltering Approach to Acoustic Echo Cancellation](https://arxiv.org/abs/2012.08867), EUSIPCO 2021

# GAN-related
## [Generative Adversarial Network-Based Postfilter for STFT Spectrograms](https://www.isca-speech.org/archive_v0/Interspeech_2017/pdfs/0962.PDF), Interspeech 2017

Generated spectra typically lack the fine structures that are close to those of the true data. Propose a GAN-based postfilter that is implicitly optimized to match the true feature distribution in adversarial learning.

GAN cannot be easily trained for very high-dimensional data such as STFT spectra. Thus take divide-and-concatenate strategy: first divide the spectrograms into multiple freq bands with overlap, reconstruct the individual bands using the GAN-based postfilter trained for each band, and connect th bands with overlap.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202205/20220613095620.png)

## [Wavecyclegan2: Time-domain neural post-filter for speech waveform generation](https://arxiv.org/abs/1904.02892)

# Loss
## [Improving Perceptual Quality by Phone-Fortified Perceptual Loss Using Wasserstein Distance for Speech Enhancement](https://www.isca-speech.org/archive/pdfs/interspeech_2021/hsieh21_interspeech.pdf), Interspeech 2021

利用音素相关的信息计算 enhanced speech 和 clean speech 之间的loss (PFPL)


# Other
## [Residual Echo and Noise Cancellation with Feature Attention Module and Multi-Domain Loss Function]()
Multiple inputs (features extracted from far-end reference and the echo estimated by the Linear Adaptive Filter) are weighted by a feature attention module.

![](https://tva1.sinaimg.cn/large/e6c9d24ely1h36paadyagj20ll0c20tr.jpg)

加入 feature attention module 来更好地融合远端参考信号、LAF输出的 $E(k,f)$ 和 估计的线性回声 $C(k,f)$ 三种输入，而非简单的拼接。

## [Deep Learning-Based Joint Control of Acoustic Echo Cancellation, Beamforming and Postfiltering](https://arxiv.org/abs/2203.01793)
如题

