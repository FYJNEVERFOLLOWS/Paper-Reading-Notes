# A Time-domain Unsupervised Learning Based Sound Source Localization Method阅读笔记

# Abstract & Conclusion

Auto-encoder neural networks are adopted so that some operation like time-delay compensation can be removed. A training strategy based on the multi-task learning and acoustic transfer function is proposed as well, called joint training of alternating and splitting. Exps show that the proposed method can learn the transmission characteristics, including the change of time delay and intensity.

# Introduction

The localization cues used in the SSL include the arrival time difference and the arrival intensity difference between the multi-channel sound signals received by microphone array.

The differences of NN-based methods are mainly reflected in the differences of input features, output classes and the structure of DNNs, which have poor robustness since it is difficult to learn universal mapping. What's worse, some artificial features have to be designed and extracted in advance like phase spectrum or magnitude spectrum.

Therefore, we proposed a method based on time-domain unsupervised learning from the viewpoint of signal processing, which can make full use of both time and intensity information. We adopt auto-encoder neural networks and propose an alternating training method based on the idea of multi-task learning, which can improve the localization and generalization performance.

# Proposed Method

## A. Modeling

In an ideal environment, for a sound source, the sound signal received by microphone array can be represented as
$$
X_m=S\times ATF_m
$$
where $X_m$ represents the frequency-domain representation of the sound signal received by the $m$-th microphone in the freq domain, $S$ is the freq-domain representation of the sound source signal, $ATF_m$ represents the acoustic transfer function corresponding to the transmission path from the sound source to the $m$-th microphone.

The received signal can recover the sound source signal through the inverse filtering operation:
$$
S=X_m \times ATF_m^{-1}
$$
The received signal can be converted to the sound source signal by inverse filtering and then be restored to original received signal by filtering, namely
$$
X_m\times ATF_m^{-1}\times ATF_m=X_m
$$
So AE neural networks can be used to model the inverse filtering operation and the filtering operation of ATF by encoder and decoder.

For the $m$-th microphone channel of a certain location $n$, it can be expressed as
$$
x_m \times Encoder_{m,n} \times Decoder_{m,n} = x_m
$$
where $x_m$ represents the time-domain signal received by the $m$-th microphone.

![image-20220503202735919](https://tva1.sinaimg.cn/large/e6c9d24ely1h1vhm0uwhlj20i80f7abd.jpg)

## B. Framework

In the localization phase, we only need to use the encoder layer of AE to recover the sound source signal, while the decoder layer only helps with training the AE models. 

![image-20220503203412558](https://tva1.sinaimg.cn/large/e6c9d24ely1h1vhtqm2xzj20kz0lvmzs.jpg)

Each step during the localization phase is detailed as follows.

**Input.** The input is multi-channel frame-level time-domain received signals. The input is $M$ frames with $M\times L$ sampling points. ($M$ microphones and the sampling points size in a frame is $L$)

**Encoder operation.** The encoder model is the encoder layer of the AE neural networks, which does not involve the decoder layer.

$\hat{s}_{Encoder_{m,n}}(t)$ represents the estimated sound source signal using the $Encoder_{m,n}$, that is the output of $Encoder_{m,n}$.

**Consistency checking.** Consistency checking is to calculate the cross-correlation coefficient of the sound source signals in pairs, and then sum all the correlation coefficients, and finally obtain the sum of cross-correlation coefficients (Scorr) between estimated multi-channel sound source signals to check the consistency of the estimated multi-channel sound source signal.

The proposed method will calculate the Scorr for each location respectively and $N$ Scorr value should be obtained in this step.
$$
Scorr_{Encoder_n}=\sum_{k=1}^M\sum_{l=k+1}^MCorr(\hat{s}_{Encoder_{k,n}}(t),\hat{s}_{Encoder_{l,n}}(t))
$$
where $Corr(\cdot,\cdot)$ is the cross-correlation function and $Scorr_{Encoder_n}$ denotes the sum of cross-correlation coefficients for location $n$.

**Location estimation.** If a candidate location during scanning is consistent with the real location, the sum of correlation coefficients would be the largest one because the multi-channel estimated sound source signals would be consistent.

Therefore, the estimated sound source location $\hat{n}$ can be represented as

$\hat{n}=\mathop{argmax}\limits_n\ Scorr_{Encoder_n},n=1,...,N.$

If there are $K$ sound sources, the location of $K$ sound sources can be determined by calculating the highest $K$ peaks.

## C. Training

For AE, the unsupervised training is prone to multiple solutions in the case of less constraints, that is, the middle hidden layer is not necessarily the estimated sound sources, but also may be the high-dimensional feature of the source signal after some transformation.

In order to solve this problem, the training method based on alternating and splitting is adopted to constrain the multi-channel estimated signals to be consistent, that is, the consistency between the multi-channel estimated signals is high.

### 1) Alternating training

### 2) Splitting training

![image-20220503232515803](https://tva1.sinaimg.cn/large/e6c9d24ely1h1vmrqeggcj20mg0mtdj2.jpg)

![image-20220503232551328](https://tva1.sinaimg.cn/large/e6c9d24ely1h1vmsbveglj20m30jv419.jpg)