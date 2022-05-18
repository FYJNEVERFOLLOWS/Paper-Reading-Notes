#! https://zhuanlan.zhihu.com/p/447493389
# DOA estimation for multiple sound sources using CRNN阅读笔记

# Abstract & Conclusion

Using the magnitudes and phases of the spectrograms of all the channels as input, generating a spatial pseudo-spectrum (SPS).

# Introduction

MUSIC requires a good estimate of the number of active sources; suffers at low SNR.

DOAnet learns the number of sources from the input data, generates high precision DOA estimates and is robust to reverberation.

6 differences between DNN-based approaches and the proposed method:

a) Most focused on azimuth estimation (at most 2-D coordinates). We estimate both azi and ele for the DOA by sampling the unit sphere uniformly and predicting the probability of sound source at each direction.

b) Most focused on the estimation of a single DOA at every time frame. (Except for some estimated up to 2 sources). We do not limit the number of sources.

c) Past works were evaluated with different array geometries making comparison difficult. We evaluate the method using real spherical harmonic input signals

d) CRNN compared to MLP, CNN

e) Most used inter-channel features such as GCC-PHAT, eigen-decomposition of the spatial covariance matrix, inter-channel time delay (ITD), inter-channel level differences (ILD). We use both the magnitude and the phase component.

f) Previous method were evaluated on speech recordings. We extend them to a larger variety of sound events.

# Method

The DOAnet takes multichannel audio as the input and first extracts the spectrograms of all the channels.

The phases and the magnitudes of the spectrograms are mapped using a CRNN to two outputs sequentially. The first output, spatial pseudo-spectrum (SPS) is generated as a regression task, followed by the DOA estimates as a classification task.

**A. Feature extraction**

sr: 44.1kHz

DFT: win2048, hop1024, keep 1024 postive freq bins

$L$ frames of features, each containing 1024 magnitude and phase values of the DFT extracted in all the $C$ channels. $L \times 1024 \times 2C$ 3-D tensor as the input. (2C results from ordering the mag of all channels first, followed by the phase)

We use a seq len $L$ of 100 (=2s) in this work.

**B. DOA estimation network (DOAnet)**

![image-20211116104736604](https://tva1.sinaimg.cn/large/008i3skNly1gwgsrmwgotj30mi0gz77c.jpg)

BN before ReLu after CNN layer

CNN output $L\times2\times N_C$, $N_c$ for the number of CNN filters in the last CNN layer, then reshaped to $L \times 2 N_C$.

The RNN output is mapped to the first output, the SPS, in regression manner using FC layers with linear activation.

10 degrees resolution

614 sampled directions

a subset of 432 directions is used for DOA (ele range -60~60, 12 * 36)

The SPS is further mapped to DOA estimates–the final output of the proposed method–using a similar CRNN network as above with two minor architectural changes.

Each node in this output layer represents a direction in 2-D polar space. During testing, the probabilities at these nodes are thresholded with a value of 0.5.



Training params:

1000 epochs

Adam optimizer

MSE for SPS (DOAnet's estimation with respect to MUSIC SPS)

binary cross entropy loss for DOA output

The sum of the two losses was used for BP

# Evaluation

**A.Dataset**

Six datasets

$Ox$: x sources, x=1,2,3

$OxA$: anechoic

$OxR$: reverberant

Each of these datasets has 3 cross-validation splits with 240 recordings for training and 60 for testing.

During cross-validation, randomly chose disjoint sets of 16 and 4 egs for training and testing, amounting to 176 egs and 44 (11 sound event classes).

**B. Baseline**

MUSIC

**C. Metric**

SPS is evaluted with respect to the baseline MUSIC estimated ground truth.
$$
\begin{aligned}
&S N R=10 \log _{10}\left(\sum_{\phi} \sum_{\lambda} S_{G T}(\phi, \lambda)^{2} / \sum_{\phi} \sum_{\lambda}\left(S_{E}(\phi, \lambda)-\right.\right.\left.\left.S_{G T}(\phi, \lambda)\right)^{2}\right) .
\end{aligned}
$$


The DOA metric:
$$
\sigma=\arccos \left(\sin \phi_{E} \sin \phi_{G T}+\right.\left.\cos \phi_{E} \cos \phi_{G T} \cos \left(\lambda_{G T}-\lambda_{E}\right)\right) \cdot 180.0 / \pi
$$
E for estimated, GT for ground truth, phi for azimuth, lamda for elevation.



**D. Evaluation procedure**

![image-20211116213804330](https://tva1.sinaimg.cn/large/008i3skNly1gwhbkg6cj0j30ly0dg40d.jpg)

Unknown  num_sources时，取声源数目一致的最小 error (Hungarian algorithm)

Only 42.7% of the estimated frames had the correct number of DOA predictions.

# Results and Discussion

