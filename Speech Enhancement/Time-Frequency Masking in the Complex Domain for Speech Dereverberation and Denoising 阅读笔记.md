#! https://zhuanlan.zhihu.com/p/516722805
# [复数域语音增强] Time-Frequency Masking in the Complex Domain for Speech Dereverberation and Denoising 阅读笔记

# Abstract

This paper addresses monaural speech separation in reverberant and noisy environments. 

Enhance the magnitude and phase by performing separation with an estimate of the complex ideal ratio mask.



# Introduction

Speaker can hear not only the sound that directly reaches their ears, but also reflections off the walls, ceiling and furniture. 

These reflections, termed reverberation, are altered versions of the original speech. 

Reverberant speech consists of 3 components: the direct sound (anechoic part corresponding to the first wavefront), early reflections (arrive up to 50ms after the direct sound) and late reflections (come anytime thereafter). 

Reverberation combined with additive noise can be detrimental to the speech intelligibility of normal hearing listeners. 

A solution for removing reverberation and noise would be beneficial for a variety of speech processing tasks.

Weighted Prediction Error is an unsupervised approach that operates in the complex T-F domain and uses linear prediction to shorten the RIR, which in effect removes late reverberation. But WPE does not address noise problem.

Performing T-F masking in the complex domain (cIRM) is very beneficial when dealing with background noise.

We propose to use DNNs to learn a mapping from reverberant (and noisy) speech to the cIRM.



# Notations and definitions

$$
y(t)=h_d(t)*s(t)+h_e(t)*s(t)+h_l(t)*s(t)\\
=d(t)+y_e(t)+y_l(t)
$$

$h(t)$ denotes the RIR, $s(t)$ denotes clean anechoic speech

$d(t)$ denotes the direct sound, $y_e(t)$: the early reflections, $y_l(t)$: the late reflections
$$
y_{rn}(t)=h_s(t)*s(t)+\beta h_n(t)*n(t)
$$
$y_{rn}(t)$: reverberant and noisy speech

$n(t)$: the noise at time $t$, $\beta$ controls the SNR between the reverberant noise and speech.

Our goal is to estimate the STFT of the direct sound $D$, since it is clean and anechoic.



# Algorithm Description

Propose to use a DNN to learn a spectral mapping from reverberant (and noisy) speech to the cIRM.

**A. Features**

The feature vector centered at the $k$th time frame is $k\pm p$ time frames 

**B. cIRM**
$$
\begin{aligned}
M(k, f)=& \frac{D(k, f)}{Y(k, f)} \\
=& \frac{Y_{r}(k, f) D_{r}(k, f)+Y_{i}(k, f) D_{i}(k, f)}{Y_{r}(k, f)^{2}+Y_{i}(k, f)^{2}} \\
&+j \frac{Y_{r}(k, f) D_{i}(k, f)-Y_{i}(k, f) D_{r}(k, f)}{Y_{r}(k, f)^{2}+Y_{i}(k, f)^{2}}
\end{aligned}
$$
$r$ for real component, $i$ for imaginary component
$$
M(k, f)=\frac{|D(k, f)|}{|Y(k, f)|} e^{j\left(\phi_{d}(k, f)-\phi_{y}(k, f)\right)}
$$
$\phi_d:$ the phase of the direct speech

$\phi_y:$ the phase of the reverberant observation

$\hat{D}=\hat{M}Y$, where $\hat{M}=\hat{M}_r+j\hat{M}_i$

# Discussion and Conclusion

PSM estimation is close to cIRM estimation likely due to the challenge of estimating the imaginary portion of the cIRM, which is less structured than the real component. This indicates that refinements for estimating the imaginary component should be developed.

Since the real and imaginary components of the cIRM are related, it is important that DNN can further capitalize on this relationship.

