#! https://zhuanlan.zhihu.com/p/447494486
# Robust Source Counting and DOA Estimation Using SPS and CNN阅读笔记

# Abstract & Conclusion

Propose to use a multi-task CNN network to estimate the number of sources and DOAs from short-time SPS. 

The short-time SPS contain useful DOA information with temporal information and decouple the DOA cues from other signal characteristics.

The estimated number of sources is used to select the final DOAs without using a fixed threshold.

2D CNN is a useful tool for peak detection.

# Introduction

**A. Signal Processing-Based Methods**

4 categories:

TDOA; sub-space; beamforming; histogram analysis

Sub-space methods such as MUSIC relies on the spatial correlation matrix



Under different conditions of noise, rever- beration, and different number of sources, the SPS are noisy. As a result, peak detection algorithms that rely on a pre-determined threshold for all of the cases are prone to errors, and severely affect the accuracy of DOA estimation algorithms.

Aims to mitigate this problem by adding a learning component to SP-based DOA estimation algorithms.



**B. Deep Learning-Based Methods**

![image-20211119161420671](https://tva1.sinaimg.cn/large/008i3skNly1gwkj2j6a8vj316309gdi2.jpg)



**C. Evaluation Metrics**

Estimated DOAs not equal to Reference DOAs: Hungarian algorithm or Recall



**D. The Proposed Method**

When the number of sources and the level of noise and reverberation increase, the magnitude of the SPS reduces and the local maxima are less well-defined.

A single threshold to determine the source directions for all the cases is not optimal for all cases.

We would like to combine the strengths of both SP-based and DL-based methods to robustly estimate DOAs of multiple sound sources in noisy and reverberant environments when the number of sound sources is unknown.

Propose a method that use SPS as input features to a CNN model to estimate the DOAs.

The SPS are used as input features to the CNN to reduce the tendency of the neural network to learn unwanted association between the sound classes and the directional information.

The DL-based methods for DOA estimation often formulate the problem as multi-label, multi-class classification. The num- ber of discretized DOAs is equal to the number of classes. The number of classes is often much larger than the number of active sound sources. As a result, there is an imbalance in the loss of active sources and inactive sources. We proposed to use a weighted cross-entropy loss as the DOA estimation loss to mitigate this problem.



The main contribution of the paper is a multi-task CNN network with weighted cross-entropy loss that use 2D SPS as input feature to robustly estimate DOAs of multiple sound sources in different noisy and reverberant environments even when the number of sources is unknown.

# DOA Estimation Using SPS and CNN

**A. Short-Time SPS Estimation**

Because the num of sound sources is not known in advance, to simplify the computation of MUSIC spectra, we assume the W-disjoint orthogonality (WDO) condition, in which each TF bin is dominated by a single speaker, to compute the SPS:
$$
P_{M U S I C}(\theta)=\frac{1}{a^{H}(\theta) Q_{n} Q_{n}^{H} a(\theta)}
$$
$a(\theta)$ is the steering vector and $Q_n$ is the noise subspace. 

$M$: num of mics, $M-1$: the dimension of the noise space.

WDO condition is mainly used for speech signal, does not hold for all the TF bins.



**B. DOA Estimation Algorithm Using CNN, Multi-Task Learning, and Weighted Cross-Entropy**

In this research, we estimate DOAs for each 2-second block (173 frames) of data.

![image-20211129164055637](https://tva1.sinaimg.cn/large/008i3skNly1gww418hg0qj30wa0u0jv5.jpg)

Resolution: 5°

The input SPS are normalized by a scalar mean and a scalar standard deviation that are estimated using all training SPS along both time and angular dimensions. 

The ouput prediction for the number of sources can only be one value between 0 and $N$ inclusively, where $N$ is the pre-defined maximum number of sources (4 in this study).

CE loss for ndoa_loss:
$$
n d o a_{-} l o s s=-\frac{1}{L} \sum_{l=1}^{L} \sum_{n=0}^{N} y_{n}^{l} \log \hat{y}_{n}^{l}
$$
$L$: the num of samples

$n$: the class index

$y_n^l$: 0 or 1



DOA estimation: doa_loss (CE loss)

Because the number of the active DOAs ($≤ N = 4$) is much smaller than the total number of possible DOAs K = 72, weighted binary cross-entropy loss is used to increase the recall rate:
$$
\text { doa\_loss }=-\frac{1}{L} \sum_{l-1}^{L} \sum_{k=0}^{K}\left(\alpha y_{k}^{l} \log \hat{y}_{k}^{l}+\left(1-y_{k}^{l}\right) \log \left(1-\hat{y}_{k}^{l}\right)\right)
$$
$K$: the num of DOA classes, i.e. the num of possible DOAs

$\alpha$: the positive weight (100 in this study)

$y_k^l \in [0,1]$ instead of $y_k^l \in {0,1}$ is because we use label smoothing (Gaussian Function).

The total loss of the multi-task network is:
$$
loss = \gamma ndoa\_loss + (1-\gamma)doa\_loss
$$
$\gamma$ was set to 0.1.

Use label-smoothing to encode the ground-truth DOAs to regularize the network. The gt DOA is encoded as a Gaussian with the maximum value centered at the true DOA instead of one-hot encoding:
$$
p_\theta=exp((\theta-\theta_{gt})^2/\sigma^2)
$$
$\sigma$ is the standard deviation that controls the spread of the curve. We use $\sigma = 2.5°$.

![image-20211120170307339](https://tva1.sinaimg.cn/large/008i3skNly1gwlq3lelpaj30ka0cxwey.jpg)

Use random-cutout augmentation for the SPS to reduce over-fitting. (Randomly set a rectangular area on the SPS to zeros.)



# Evaluation

A circular planar UMA-8 mic array.



**A. Datasets**

![image-20211120170624019](https://tva1.sinaimg.cn/large/008i3skNly1gwlq6ywz6jj30lj07kq3y.jpg)

**B. Evaluation Metrics**

maximal matching & recall & precison & F1

用匈牙利算法算最大匹配数对应的匹配之间的差的加和

![preview](https://pic3.zhimg.com/v2-81f21981c992bc0b5b1acf04b37ff6c2_r.jpg)
$$
\text { doa\_error }=\frac{1}{\sum_{i=1}^{L} n_{e}^{i}} \sum_{i=1}^{L} H\left(\boldsymbol{\theta}_{e}^{i}, \boldsymbol{\theta}_{g t}^{i}\right)
$$
$L$: the num of test samples

$n_e^i$: the num of estimated DOAs of sample $i$

$\theta_e^i$: the list of estimated and ground truth DOAs



# Experiments

**B. Hyper-Parameters**

sr: 44.1kHz

win: 1024, hop: 512, Hann, 1024 FFT points, 96 mel filters, GCC-PHAT length of 96 lags

The length of one input sample was 2 secs, which translated to $T=173$ frames.

Adam, 80 epochs, 0.001 lr for the first 50 epochs and reduced by 10% for each subsequent epoch.



# Results and Discussion

![image-20211121101141674](https://tva1.sinaimg.cn/large/008i3skNly1gwmjtt5sp7j31040fpdk6.jpg)

T: if a threshold was required by the algorithm to determine DOAs

SS: single-source TF bin selection

MP: magnitude and phase spectrogram

GCC: GCC-PHAT

LG: a combination of log-mel spectrogram and GCC-PHAT as input features



The magnitude and phase spectrogram, log-mel spectrogram, and GCC-PHAT were normalized by their mean and standard deviation vector along the frequency dimension for each channel separately.

![image-20211121102736083](https://tva1.sinaimg.cn/large/008i3skNly1gwmkac1v37j30mm0drwh7.jpg)
