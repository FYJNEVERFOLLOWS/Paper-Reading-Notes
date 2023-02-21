#! https://zhuanlan.zhihu.com/p/497216650
# Robust DOA Estimation Based on Convolutional Neural Network and Time-Frequency Masking阅读笔记

# Abstract & Conclusion

Inspired by the success of time-frequency masking in speech enhancement and speech separation, this paper proposes new methods to better utilize time-frequency masking in convolution neural network to improve the robustness of localization.

First a mask estimation network is developed to assist DOA estimation by either appending or multiplying the estimated masks to the original input feature. Then we further propose a multi-task learning architecture to optimize the mask and DOA estimation networks jointly, and two modes are designed and compared.

We propose three DOA estimation methods based on CNN and T-F masking, including integrating the T-F mask module for DOA separately and jointly optimizing T-F mask model with DOA simultaneously.

Further work will consider different input features for mask and DOA estimation and a better metric for mask estimation.

# Intro

This work aims to improve the robustness by eliminating most noise-dominated T-F bins in the feature to minimize the effects of noises and reverberation. 



# Problem Description

The phase information is the essence in the DOA estimation task.

# CNN Based DOA Estimation with T-F Masking

## 3.1. CNN for DOA

In the CNN based framework, DOA estimation is generally formulated as an $I$-class classification problem, where $I$ denotes  the number of classes. Phase-related features are fed into the CNN and a mapping from input features to the corresponding DOA label is learned.

![image-20220407164636266](https://tva1.sinaimg.cn/large/e6c9d24ely1h1194yfff7j214y0h80v4.jpg)

The input vector is the phase component of the STFT coefficient of the received signal at each microphone instead of explicitly extracted features. ($M$ denotes the number of mics, $K$ denotes the total num of freq bins)

The output is an $I \times 1$ vector indicating the posterior probabilities of the $I=72$ DOA classes.

The basic CNN usually needs data pre-processing such as VAD to eliminate non-speech frames, which may not be accurate and cannot eliminate the effects of noise in different freq bins. So we propose three methods to improve the performance of the CNN-based method.

## 3.2. CNN with Ideal Ratio Mask for DOA

![image-20220407165740514](https://tva1.sinaimg.cn/large/e6c9d24ely1h119gf662jj215o0ow77l.jpg)

Mask model and DOA model are built separately: First we train the mask estimation network to derive a magnitude-related mask which represents the probability of each T-F bin being dominated by the target speech signal. Then we enhance the input features with the estimated mask and train the DOA estimation network with these new features. (Showed in Fig.2)

To enhance the input feature, we can simply append the mask to the 6-channel input as an additional feature. We also tried multiplying the input by the mask to minimize the effects of noise-dominated T-F bins, and thus the mask is regarded as the weight of each T-F bin in input features.

The mask estimation network is a regression model that maps noisy log-magnitude feature to the corresponding clean mask. The input vector consists of 11 consecutive frames (5 preceding and 5 following the current frame) of the log-magnitude spectrum of the received signal at each mic. The output is the estimated soft mask of the current frame. We compute the target mask label for each frame as follows:
$$
IRM=\frac{S^2(t,f)}{S^2(t,f)+N^2(t,f)}
$$
where $S(t,f)$ and $N(t,f)$ denote the magnitude spectrum of clean speech signal and noise signal at the $t$-th time frame and the $f$-th frequency bin respectively. MSE is used for trainning the mask estimation network.

## 3.3. Multi-task learning for DOA estimation

### 3.3.1. Standard multi-task learning

![image-20220407222910382](https://tva1.sinaimg.cn/large/e6c9d24ely1h11j1fj4vtj22i00m8460.jpg)

Two inputs and two outputs:

T-F mask network's input: log-magnitude spectrum

DOA network's input: the phase spectrum which is multiplied by the predicted mask

Two outputs are the estimated T-F mask and the DOA classification

Loss is a combination of the MSE loss for mask estimation network and the CE loss for DOA estimation network:
$$
\mathcal{L}_{multi}=\alpha\mathcal{L}_{MSE}+(1-\alpha)\mathcal{L}_{CE}
$$
where $\alpha$ is a constant and set to 0.01 in our experiments.

### 3.3.2. Pseudo multi-task learning

In this strategy, we use the DOA classification loss to update the entire network by setting $\alpha$ to 0 as we only care about the DOA estimation task.

We remove the explicit constraints on the mask estimation output so that the network can learn a mask that best matches the DOA estimation task.



# Experimental results

## 4.1. Experimental setup

The input log-magnitude features are all normalized to [-1,1]