#! https://zhuanlan.zhihu.com/p/562422705
# Beam-TasNet: Time-domain Audio Separation Network Meets Frequency-domain Beamformer 阅读笔记

# Abstract
Acoustic beamforming using a microphone array plays an important role in the construction of high-performance ASR systems. TasNet achieves remarkable speech separation performance. In light of these two recent trends, the question of whether TasNet can benefit from beamforming to achieve high ASR performance in overlapping speech conditions naturally arises.

Proposes Beam-TasNet, which combines TasNet with the frequency-domain beamformer MVDR through spatial covariance computation to achieve better ASR performance.

# 1. Intro
Multi-channel linear filtering (beamforming) helps the construction of high-performance ASR systems for noisy and multi-talker ASR tasks.

The neural network-supported mask-based beamformer is one of the most successful beamforming schemes and used as a key component of robust ASR system.

TasNet subsequent work [A comprehensive study of speech separation: spectrogram vs waveform separation] utilize multi-channel signals as input but generates the output signal directly as the output of TasNet, which is a nonlinear transformation that may include processing artifacts. Such artifacts could limit ASR performance improvement.

Motivated by the success of the neural network-supported beamformer, we combine TasNet with the well-studied freq-domain MVDR beamformer. The proposed method applies TasNet separately to each of the multi-channel inputs and uses the estimated time-domain signals to compute the Spatial Covariance matrices for the freq-domain MVDR.

# 2. Overview of TasNet
## 2.1. Single-channel TasNet
## 2.2. Multi-channel TasNet
$$
\left\{\hat{\mathbf{x}}_{i, c}\right\}_{i=1}^I=\operatorname{MC}-\operatorname{Tas} \operatorname{Net}\left(\mathbf{y}_c, \mathbf{y}_{c_1^{\prime}}, \cdots, \mathbf{y}_{c_{C^{\prime}}^{\prime}}\right)
$$
$c$: the main channel index

$c_1^{\prime},...,c_{C^{\prime}}^{\prime}$: the auxiliary channel indices

$C^{\prime}$: the number of auxiliary channels

$I$: the number of output sources

# 3. Proposed Beam-TasNet

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220907213853.png)

## 3.1. Freq-domain beamforming
$\mathbf{Y}_{t,f}$: the observed $C$-channel mixture 
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220907214939.png)

## 3.2. Spatial covariance estimation using TasNet outputs
By separately applying TasNet for each channel (while changing the main and auxiliary channels for MC-TasNet), we obtain $C$-channel TasNet outputs corresponding to the separated waveforms for each source $\{\hat{\mathbf{x}}_{i=1, c}\}^C_{c=1}, ..., \{\hat{\mathbf{x}}_{i=I, c}\}^C_{c=1}$.

In sig-MVDR scheme, we calculate the SC matrices directly from the TasNet outputs (i.e., without computing TF masks) as:

$$
\begin{aligned}
&\boldsymbol{\Phi}_f^{\mathrm{S}_i}=\frac{1}{T} \sum_{t=1}^T \hat{\mathbf{X}}_{i, t, f} \hat{\mathbf{X}}_{i, t, f}^{\mathrm{H}}, \\
&\boldsymbol{\Phi}_f^{\mathrm{N}_i}=\frac{1}{T} \sum_{t=1}^T\left(\mathbf{Y}_{t, f}-\hat{\mathbf{X}}_{i, t, f}\right)\left(\mathbf{Y}_{t, f}-\hat{\mathbf{X}}_{i, t, f}\right)^{\mathrm{H}},
\end{aligned}
$$

where $\hat{\mathbf{X}}_{i, t, f}$ is the $C$-channel STFT computed from the $i$-th source signal $\{\hat{\mathbf{x}}_{i, c}\}^C_{c=1}$.

In mask-MVDR scheme, the TF masks for speech signals are calculated as the ratio of the magnitudes between the TasNet outputs and the input mixtures. 

## 3.3. Inter-channel permutation problem
The order of sources at the output may be different for each channel (Fig. 1). [The order of the TasNet outputs is not guaranteed to be properly aligned for each channel.]

Thus introduce an inter-channel permutation solver based on the cross correlation function $\mathrm{xcorr}(\cdot)$:
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220907221702.png)
## 3.4. training objective
Adopt SNR

To estimate the SC matrix for the noise signal, the TasNet outputs should retain the scale info of each source.

# 4. Exp
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220907222049.png)


