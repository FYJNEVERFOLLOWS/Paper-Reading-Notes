# A Gentle Summary of Postfilter-related Work
近年来基于深度学习方法的 postfilter 的工作大致有以下几种：
1. 通过训练目标等设计 encoder-decoder 结构中的 decoder 起到 postfilter 的作用 [1]；
2. 先进行预分离，再通过基于 NN 的 postfilter 进行进一步分离 [2,3]；
3. 在 AEC 的任务中，讲传统的 postfilter 模块替换成 NN 的 [4,5]；
4. 添加额外网络进行后处理 [6]；
5. 用 GAN 的判别器对增强后的语谱进行后处理 [7,8]。
   
下面是这些文章的详细介绍

# NN-based Postfilter (Decoder)
## [Inplace Gated Convolutional Recurrent Neural Network for Dual-Channel Speech Enhancement](https://www.isca-speech.org/archive/pdfs/interspeech_2021/liu21f_interspeech.pdf), Interspeech 2021
Decoder: signal filtering & reconstruction （两个 encoder 分别对 mag 和 phase 进行 masking 和 mapping）

## [End-to-End Post-Filter for Speech Separation With Deep Attention Fusion Features](https://ieeexplore.ieee.org/abstract/document/9043689), TASLP 2020
先在时频域预分离，然后将混合语音和预分离的语音同时作为输入，通过 一维卷积和 attention 进行特征融合，融合的特征送入 TCN-based postfilter

## [Dual-Path Filter Network: Speaker-Aware Modeling for Speech Separation](https://www.isca-speech.org/archive/pdfs/interspeech_2021/wang21x_interspeech.pdf), Interspeech 2021
先预分离，再根据 speaker 信息进一步分离
[笔记](https://zhuanlan.zhihu.com/p/530248603)

## [$Y^2$-Net FCRN for Acoustic Echo and Noise Suppression](https://arxiv.org/abs/2103.17189), Interspeech 2021
用两个全卷积循环网络 FCRN，首先是 AEC 模块估计回声，再用后置滤波模块进行残留回声抑制。

## [Bandwidth-Scalable Fully Mask-Based Deep FCRN Acoustic Echo Cancellation and Postfiltering](https://arxiv.org/abs/2205.04276)
Follow $Y^2$-Net 的工作，增加了频带宽度扩展 Bandwidth Extension。

## [SFSRNet: Super-Resolution for Single-Channel Audio Source Separation](https://www.aaai.org/AAAI22Papers/AAAI-1535.RixenJ.pdf), AAAI 2022
decoder 后面接上超分网络 + 渐进学习恢复降采样带来的信息损失
[笔记](https://zhuanlan.zhihu.com/p/532595774)

# GAN-related
## [Generative Adversarial Network-Based Postfilter for STFT Spectrograms](https://www.isca-speech.org/archive_v0/Interspeech_2017/pdfs/0962.PDF), Interspeech 2017

Generated spectra typically lack the fine structures that are close to those of the true data. Propose a GAN-based postfilter that is implicitly optimized to match the true feature distribution in adversarial learning.

GAN cannot be easily trained for very high-dimensional data such as STFT spectra. Thus take divide-and-concatenate strategy: first divide the spectrograms into multiple freq bands with overlap, reconstruct the individual bands using the GAN-based postfilter trained for each band, and connect th bands with overlap.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202206/20220613095620.png)

## [Wavecyclegan2: Time-domain neural post-filter for speech waveform generation](https://arxiv.org/abs/1904.02892)

# Conventional
## [Nonlinear Spatial Filtering in Multichannel Speech Enhancement](https://arxiv.org/abs/2104.11033), TASLP 2021

## [A Synergistic Kalman- and Deep Postfiltering Approach to Acoustic Echo Cancellation](https://arxiv.org/abs/2012.08867), EUSIPCO 2021


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



Filterbank design for end-to-end speech separation, Manuel Pariente et al., ICASSP 2020

