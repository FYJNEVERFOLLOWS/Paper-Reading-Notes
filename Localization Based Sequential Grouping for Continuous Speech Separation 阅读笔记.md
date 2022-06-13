# [多通道语音分离] Localization Based Sequential Grouping for Continuous Speech Separation 阅读笔记

# Abstract
This study investigates robust speaker localization for continuous speech separation and speaker diarization, where we use speaker directions to group non-contiguous segments of the same speaker. Assuming that speakers do not move and are located in different directions, the DOA info provides an informative cue for accurate sequential grouping and speaker diarization.

Our system is block-online: given a block of frames with at most two speakers, we apply a two-speaker separation model to separate (and enhance) the speakers, estimate the DOA of each separated speaker, and group the separation results across blocks based on the DOA estimates.

# Introduction
Investigate the use of spatial information for speaker diarization.

Many studies extract DOA information directly from noisy-reverberant multi-speaker mixtures (or from enhanced or separated mixtures that are not sufficiently accurate), which usually lead to inaccurate time-delay features and DOA estimation.

# Proposed Algorithms
Our system is block-online. In each short processing block, we assume that there are at most $C$ (=2 in this study) speakers. Within each block,

$$
\begin{aligned}
\boldsymbol{Y}(t, f) &=\sum_{c=1}^{C} \boldsymbol{X}(c, t, f)+\boldsymbol{N}(t, f) \\
&=\sum_{c=1}^{C}(\boldsymbol{S}(c, t, f)+\boldsymbol{H}(c, t, f))+\boldsymbol{N}(t, f)
\end{aligned}
$$

$Y(t,f)$: the complex STFT vectors of the received mixture

$S$ for direct-path signal

$H$ for early reflections plus late reverberation

$N$ for reverberant noise

$Y_q$ denotes the spectrogram of the mixture at mic $q$, $S_q(c)$ denotes that of speaker $c$.

Aim at estimating $S_q(c)$ for each source at the reference mic based on the multi-channel input $Y$.

**At each block, we perform separation (and enhancement) and count the number of speakers at each frame. Based on the separation and counting results, we localize each speaker. Next, we group the separation results across blocks based on the localization results, and feed the grouped separation results into an ASR backend for recognition.**

## 2.1. MISO-BF-MISO for Block-Online Separation

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202205/20220606154935.png)

$MISO_1$: [Multi-Microphone Complex Spectral Mapping for Speech Dereverberation]

$MISO_2$: [Multi-Microphone Complex Spectral Mapping for Utterance-Wise  and Continuous Speaker Separation]

Predict the real and imaginary (RI) components of target speech at a reference mic from the RI components of the stacked multi-channel input signals.

The first network is trained using utt-wise PIT to estimate the direct-path signal of each speaker at each mic, denoted as $\hat{S}_q^{(1)}(c)$, where the superscript indicates that it is produced by the first DNN. The target estimates are then utilized to compute spatial covariance matrices for MVDR beamforming. The second MISO network takes in the outputs of the first network and the beamforming results, denoted as $\widehat{BF}(c)$, to enhance each target speaker. The output is denoted as $\hat{S}_q^{(2)}(c)$.

## 2.2. Frame-wise Speaker Counting
Train a MISO based speaker counting network to predict the number of speakers at each frame. The input features are the stacked RI components of the mixture, plus the magnitude features at the reference mic.

Perform 3-class classification for frame-wise speaker counting.

## 2.3. Mask-Weighted GCC-PHAT for DOA Estimation
The key idea of [Robust Speaker Localization Guided by Deep Learning Based Time-Frequency Masking] is to use a DNN to accurately estimate IRM to utilize cleaner phase information of T-F units for accurate localization.

$$
\begin{aligned}
&G_{p, p^{\prime}}(t, f, k) \\
&\quad=\operatorname{Real}\left\{\frac{Y_{p}(t, f) Y_{p^{\prime}}(t, f)^{\mathrm{H}}}{\left|Y_{p}(t, f)\right| \mid Y_{p^{\prime}}(t, f)^{\mathrm{H} \mid}} e^{-j 2 \pi \frac{f}{N} f_{s} \tau_{p, p^{\prime}}(k)}\right\} \\
&=\cos \left(\angle Y_{p}(t, f)-\angle Y_{p^{\prime}}(t, f)-2 \pi \frac{f}{N} f_{s} \tau_{p, p^{\prime}}(k)\right)
\end{aligned}
$$
GCC-PHAT coefficients essentially measure the cosine distance between the observed IPD, $\angle Y_{p}(t, f)-\angle Y_{p^{\prime}}(t, f)$, and hypothesized IPD, $2 \pi \frac{f}{N} f_{s} \tau_{p, p^{\prime}}(k)$, at each T-F unit.

The GCC-PHAT coefficients are summated over all the microphone pairs and over all the T-F units within the block. The direction $k$ producing the highest summation is considered as the estimated direction.

## 2.4. DOA Based Sequential Grouping
...

# Conclusion
Proposed for continuous speech separation a sequential grouping technique using deep learning based speaker separation and localization.