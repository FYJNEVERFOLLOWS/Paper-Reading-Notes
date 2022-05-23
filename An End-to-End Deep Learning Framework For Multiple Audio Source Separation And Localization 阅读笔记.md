#! https://zhuanlan.zhihu.com/p/518440711
# [时域分离+DOA] ICASSP 2022 阅读笔记.md

# An End-to-End Deep Learning Framework For Multiple Audio Source Separation And Localization 阅读笔记
# Abstract
Sound source separation and localization for situational awareness enables a wide range of applications such as hearing enhancement and audio beam-forming.
The proposed framework jointly estimates the separated sources and their TDOA at different microphones, then it obtains the DOA for each source. A new structure to reconstruct the mixed signal is introduced for joint optimization of source separation and TDOA estimation. In addition, a discriminator network is added during the training phase to further improve the separation quality.

# Introduction
In the case of multi-sources as in the *cocktail party problem*, a good source separation strategy is the premise for precise DOA estimation.

Our approach is motivated by the observation that, for the multi-source localization problem, some intermediate information such as the separated source signals and TDOA between mics can be explicitly obtained and utilized to improve the overall system performance.

In our approach, separated source signals and their TDOA on mics are jointly estimated, then the DOA of each source is estimated from TDOA.

# Method
Proposed model consists of a source separation network, TDOA estimation network, and DOA estimation network.

![Framework diagram](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202205/20220521165932.png)

TDOA info is estimated from the multi-channel mixtures, then sent to the DOA estimation network for localizing each source. The mixture is reconstructed using the separated sources and estimated TDOA, and we evaluate the similarity loss between the reconstructed $\hat{x}$ and original mixture $x$ for joint optimization of source separation and TDOA estimation. In the meantime, a discriminator is added to improve the quality of separated signals.

## 2.1. Source separation
The SS network extracts each audio source from multi-channel mixture. 

The separated audio quality is typically measured by scale-invariant signal to noise ratio (SI-SNR), defined by
$$
\operatorname{SI-SNR}(s, \hat{s})=10 \log _{10} \frac{\|\alpha s\|^{2}}{\|\alpha s-\hat{s}\|^{2}}
$$
The separation network is trained to minimize the negative permutation-invariant SI-SNR, defined as
$$
\mathcal{L}(s^*,\hat{s})=-\text{SI-SNR}(s^*,\hat{s})
$$
where $s^*$ denotes the permutation of the sources that maximizes SI-SNR.

## 2.2. TDOA estimation
Propose to simultaneously discriminate and localize $n$ sound sources using the TDOA estimated with an array of $K$ microphones. The TDOA $\Delta T_{ij}$ is defined as
$$
\Delta T_{ij} = \frac{f_s}{c}(||l_s-l_i||-||l_s-l_j||)
$$
where $l_s$ is the location (coordinate) of the source

$l_i$ / $l_j$: the locations of the microphone $i$ / $j$

$f_s$: sampling frequency, $c$: the speed of sound

Only $K-1$ of total $K(K-1)/2$ different TDOAs are independent. So we choose to use only $\Delta T_{1j}$ values for DOA estimation.

Instead of estimating TDOA by identifying the peak of the cross-correlation of two signals, propose to use a TDOA estimation network, which can be easily integrated and jointly trained with the source separation network to build an end-to-end system.

Each source signal received at microphone 1 is treated as the reference (non-shifted version) in the source separation stage. TDOA estimator uses the estimated reference signal together with the mixtures received at other microphones to estimate the TDOA between mic pairs regarding the same source.

We treat TDOA estimation as a classification problem where each class represents a possible TDOA in terms of sample index (as the TDOA is discretized due to audio signal sampling and its maximum is limited by the configuration of the mic array).

We use the original source signals for the initial training of the TDOA estimation network. At this point, we train the source separation network and TDOA estimation network can be independently trained with their own loss functions. Then, we combine source separation and TDOA estimation networks for joint end-to-end training to improve both separation quality and TDOA estimation accuracy. For that, we define a similarity loss between the original mixture $x$ and the reconstructed mixture $\hat{x}$ obtained by applying the estimated TDOA information to the separated source signals.
$$
\mathcal{L}_{\mathrm{sm}}(x, \hat{x})=-\frac{1}{K} \sum_{i=1}^{K}\left\langle x_{i}^{T}, \hat{x}_{i}\right\rangle
$$
$K$ is total number of channels (microphones) and $\left\langle\cdot,\cdot\right\rangle$ denotes inner product.

This loss is designed to make the reconstructed multi-channel mixture as close as possible to the original one. The main issue of applying this similarity loss to network training is the non-differentiable TDOA time-shift operation to reconstruct mixture signals. We mitigate this issue by treating softmax of the TDOA estimation output vector $y_i$ as the channel impulse response and convolving it with the separated signal $\hat{s}_j$ to obtain a time shifted version $\widetilde{s}_j$ of source $j$. Then the reconstructed mixture $\hat{x}_i$ of channel $i$ can be obtained by
$$
\hat{x}_{i}=\sum_{j=1}^{N} \tilde{s}_{j}=\sum_{j=1}^{N} \operatorname{softmax}\left(y_{j}\right) * \hat{s}_{j}
$$
$N$ is the number of sources, $*$ is convolution. This technique allows end-to-end training of source separation and TDOA estimation networks through back-propagation.

Inspired by the success of GANs, we adopt a discriminator network in our framework to distinguish estimated/separated source signals (fake samples) from original source signals (real samples). This discriminator is only used during training to improve the source separation quality.

The total loss including the subjective discriminator loss, separation loss, TDOA estimation loss, and reconstruction loss is defined as
$$
\begin{aligned}
&\mathcal{L}=\mathcal{L}_{\text {sep }}\left(s^{*}, \hat{s}\right)+\mathcal{L}_{\text {TDOA }}+\alpha \cdot \mathcal{L}_{\mathrm{sm}}(x, \hat{x})\\
&+\beta \cdot \mathbb{E}_{\hat{s}}(\log (1-D(\hat{s}))),
\end{aligned}
$$
$\mathcal{L}_{\text {TDOA }}$: the CE loss for TDOA classification

$D(\hat{s})$: the discriminator output (estimated probability that $\hat{s}$ is real)

$\alpha=1$ and $\beta=0.01$ are weights for loss terms

## 2.3. DOA estimation
The DOA estimator takes the estimated TDOA of each source as the input to obtain the azimuth angle of sources regarding the mic array. A simple multilayer perception model is sufficient for this regression problem.

# Experiment Setup
16kHz, $K=4$ mics placed in a square shape with 0.2-meter

Source-to-array distance: [1, 3] m

azimuth: [0, 180]

Generate our own datasets from LibriSpeech

Mixturing is similar to LibriMix

Remove idle periods greater than 0.5 seconds.

$N=3\ \text{and}\ 4$ sources

Audio length is set to 2 s in all experiments.

2 SOTA speech separation models, SuDoRM-RF and DPTNet are adapted to serve as our source separation network.

TDOA estimation network is a 6-layer CNN with four 1D convolution layers followed by two FC layers. The maximum time-shift is less than 20 samples, thus our TDOA estimation network output has 41 (with positive and negative TDOA) classes in total. The discriminator is a CNN with four 1D convolution layers, and it is trained with BCE loss and Adam optimizer.

Jointly train for 200 epochs before adding the discriminator for alternated adversarial training.

# Results
The separation quality is improved by 1.3 – 2.7 dB with the proposed reconstruction structure and discriminator.

![Table 1](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202205/20220522222942.png)

![Table 2](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202205/20220522223012.png)

# Conclusion
Present an end-to-end DL framework for accurate source separation and localization in multi-source environments. By joint training of separation and TDOA estimation networks with a reconstruction structure and a discriminator network, the source separation quality as well as the TDOA estimation accuracy improves.