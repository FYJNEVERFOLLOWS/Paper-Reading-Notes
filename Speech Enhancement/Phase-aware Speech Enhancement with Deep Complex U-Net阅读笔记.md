#! https://zhuanlan.zhihu.com/p/497882675
# Phase-aware Speech Enhancement with Deep Complex U-Net 阅读笔记

# Abstract & Conclusion 

Most DL-based models for SE have mainly focused on estimating the mag of spectrogram while reusing the phase from noisy speech for reconstruction due to the difficulty of estimating the phase of clean speech. We tackle the phase estimation problem in three ways. First, propose Deep Complex U-Net. Second, propose a polar coordinate-wise complex-valued masking method to reflect the distribution of complex ideal ratio masks. Third, define a novel loss func, weighted source-to-distortion ratio (wSDR) loss, which is designed to directly correlate with a quantitative evaluation measure.

Propose Deep Complex U-Net which combines two models to deal with complex-valued spectrograms for speech enhancement. Designed a new complex-valued masking method optimized with a novel loss func, weighted-SDR loss.



# Intro

It has been a common practice to transform a time-domain waveform to a time-frequency representation (i.e. spectrograms) via STFT. Spectrograms are represented as complex matrices, which are normally decomposed into mag and phase components to be used in real-valued networks.

Mask-based attempts to perform the task by incorporating phase info were the proposal of the Phase-Sensitive Mask and Complex-valued Ratio Mask.

DCUnet is trained to estimate a CRM represented in polar coordinates with prior knowledge observable from ideal complex-valued masks.

Our contributions can be summarized as follows:

1. Propose **Deep Complex U-Net**, which combines the advantages of both deep complex networks and U-Net, yielding SOTA performance.
2. Design a new complex-valued masking method based on polar coordinates.
3. Propose a new loss func **weighted-SDR loss**, which directly optimizes a well known quantitative evaluation measure.



# Phase-Aware Speech Enhancement

The input mixture signal $x(n)=y(n)+z(n)\in\mathbb{R}$ is assumed to be a linear sum of the clean speech signal and noise.

$X_{t,f},Y_{t,f},Z_{t,f},\hat{Y}_{t,f}\in\mathbb{C}$ are corresponding TF representations.

The ground truth mask cIRM $M_{t,f}=Y_{t,f}/X_{t,f}$, and the estimated cRM is denoted as $\hat{M}_{t,f}$.

## 3.1 Deep Complex U-Net

![image-20220412212327168](https://tva1.sinaimg.cn/large/e6c9d24ely1h1798jz0w3j21q80eejuw.jpg)

Below is how U-Net is modified using the complex building blocks originally proposed by [Deep
complex networks](https://arxiv.org/abs/1705.09792).

**Complex-valued Building Blocks**. Complex-valued convolutional filter $W=A+iB$.

**Modifying U-Net**. Conv layers of U-Net are all replaced to complex convolutional layers. Complex batch normalization is implemented on every convolutional layer except the last layer of the network. In the encoding stage, max pooling operations are replaced with strided complex convolutional layers to prevent spatial info loss. In the decoding stage, strided complex deconvolutional operations are used to restore the size of input. 

## 3.2 Complex-valued Masking on Polar Coordinates

cRM performs a rotation on the polar coordinates, allowing to correct phase errors.
$$
\hat{Y}_{t,f}=\hat{M}_{t,f} \cdot X_{t,f} = |\hat{M}_{t,f}| \cdot |X_{t,f}| \cdot e^{i(\theta_{\hat{M}_{t,f}}+\theta_{X_{t,f}})}
$$
In this state, the real and imaginary values of the estimated cRM is unbounded. We can imagine the difficulty of optimizing from an infinite search space compared to a bounded one.

![image-20220412225924304](https://tva1.sinaimg.cn/large/e6c9d24ely1h17c0c4t0uj21ru0l47be.jpg)

## 3.3 Weighted-SDR Loss

Add a noise prediction term $loss_{SDR}(z,\hat{z})$
$$
\operatorname{loss}_{w S D R}(x, y, \hat{y}):=\alpha \operatorname{loss}_{S D R}(y, \hat{y})+(1-\alpha) \operatorname{loss}_{S D R}(z, \hat{z})
$$
where $\hat{z}=x-\hat{y}$ is the estimated noise and $\alpha=||y||^2/ (||y||^2+||z||^2)$ is the energy ratio between clean speech $y$ and noise $z$.



# Experiments

## 4.1 Experimental Setup

Noise and clean speech recordings were provided from DEMAND and Voice Bank corpus, respectively.

sr: 48kHz

SNR: 15, 10, 5, 0 dB

## 4.2 Comparison Results

![image-20220413151332934](https://tva1.sinaimg.cn/large/e6c9d24ely1h1845xjdjbj21qo0sm498.jpg)

## 4.3 Ablation Studies

