#! https://zhuanlan.zhihu.com/p/566211230
# FaSNet: Low-latency Adaptive Beamforming for Multi-microphone Audio Processing 阅读笔记

# 1. Abstract
Learning-based beamforming methods, sometimes called *neural beamformers*, have achieved signiticant improvements in both signal quality (e.g. SNR) and speech recognition (e.g.  WER). Such systems are generally non-causal and require a large context for robust estimation of inter-channel features, which is impractical in applications requiring low-latency responses.

Propose filter-and-sum network (FaSNet), a time-domain, filter-based beamforming approach suitable for low-latency scenarios. FaSNet has a two-stage system design that first learns frame-level time-domain adaptive beamforming filters for a selected reference channel, and then calculate the filters for all remaining channels. The filtered outputs at all channels are summed to generate the final output. Experiments show that FaSNet outperforms several oracle beamformers with respect to SI-SNR in reverberant speech enhancement and separation tasks.

# 2. Intro
*Beamforming*, also known as spatial filtering, is a powerful microphone array processing technique that extracts the signal-of-interest in a particular direction and reduces the effect of noise and reverberation from a multi-channel signal.

Neural beamformers can be broadly categorized into three main categories.
1. *filtering-based* (FB) approach, aims at learning a set of beamforming filters to perform filter-and-sum (FaS) beamforming in either time-domain or freq domain. FaS beamforming applies the beamforming filters to each channel and then sums them up to generate a single-channel output, within which the filters can be either fixed or adaptive depending on the model design
2. *masking-based* (MB) beamforming, estimates the FaS beamforming filters in freq domain by estimating TF masks for the sources of interest. The TF masks specify the dominance of each TF bin and are used to calculate the spatial covariance features required to obtain optimal weights for beamformers such as MVDR and GEV BF.
3. *regression-based* (RB) approach, implicitly incorporates beamforming within a neural network without explicitly generating the beamforming filters [Wave-u-net]. In this framework, the input channels are directly passed to a (convolutional) neural network and the training objective is to learn a mapping between the multi-channel inputs and the target source of interest. The beamforming operation is thus assumed to be implicitly included in the mapping function defined by the model.

Previous studies have shown that freq-domain neural beamformers significantly outnumber time-domain neural beamformers for several reasons.
1. neural beamformers are typically designed and applied to ASR tasks in which freq-domain methods are still the most common approaches.
2. freq-domain beamformers are known to be more robust and effective than time-domain beamformers in various tasks. However, in applications and devices where online, low-latency processing is required, freq-domain methods have the disadvantage that the freq resolution and the input signal length needed for a reasonable performance might result in a large, perceivable system latency.

To address the limitation of previous neural beamformers, propose FaSNet, a time-domain adaptive FaS beamforming framework suitable for real-time, low-latency applications. It consists of two stages where the first stage estimates the beamforming filter for a selected reference channel, and the second stage utilizes the output from the first stage to estimate beamforming filters for all remaining channels. The input for both stages consists of the target channel to be beamformed as well as the use of the normalized cross-correlation (NCC) between channels as the inter-channel feature. Both stages make use of TCNs for low-resource, low-latency processing. Moreover, depending on the actual task to solve, the training objective of FaSNet can be either a signal-level criterion (e.g. SNR) or ASR-level criterion (e.g. mel-spectrogram), which makes FaSNet a flexible framework for various scenarios.

# 3. FaSNet
## 3.1. Problem definition
The problem of time-domain FaS beamforming is defined as estimating a set of time-domain filters for a mic array of $N\ge2$ mics, such that the summation of the filtered signals of all microphones provides the best estimation of a signal of interest in a selected reference mic.

We first split the signals $\mathbf{x}^i$ at each mic into frames of $L$ samples with a hop size of $H$ samples
$$
\mathbf{x}_t^i=\mathbf{x}^i[t H: t H+L-1], \quad t \in \mathbb{Z}, \quad i=1, \ldots, N
$$
where $t$ is the frame index, $i$ is the index of the mic. To account for the TDOA of the signal of interest at different mics, the FaS operation is applied on a context window around frame $t$ for each mic to generate the beamformed output at frame $t$
$$
\hat{\mathbf{y}}_t=\sum_{i=1}^N \mathbf{h}_t^i \circledast \hat{\mathbf{x}}_t^i
$$
$\mathbf{y}_t$ is the beamformed signal at frame $t$

$\hat{\mathbf{x}}_t^i=\mathbf{x}^i[tH-L:tH+2L-1]$ is the context window around $\mathbf{x}_t$ for mic $i$

$\mathbf{h}^i_t$ is the beamforming filter to be learned for mic $i$

$\circledast$ represents the convolution operation

## 3.2. Reference channel processing
Use frame-level normalized cross-correlation (NCC) as the inter-channel feature. NCC feature contains both the TDOA info and the content-dependent info of the signal of interest in the reference mic and the other mics.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220911190745.png)

## 3.5. Training objectives
For tasks that take signal quality as an evaluation measure, we use SI-SNR as the training objective.

For tasks where freq-domain output is favored (e.g. ASR), use mel-spectrogram with scale-invariant mean-square-error (SI-SME) as the training objective:
$$
\begin{aligned}
\left\{\mathbf{Y}_c\right.&=\left|\operatorname{STFT}\left(\frac{\mathbf{y}_c}{\left\|\mathbf{y}_c\right\|_2}\right)\right| \\
\mathbf{Y}_c^* &=\left|\operatorname{STFT}\left(\frac{\mathbf{y}_c^*}{\left\|\mathbf{y}_c^*\right\|_2}\right)\right| \\
\mathcal{L}_{o b j} &=\frac{1}{C} \sum_{c=1}^C \operatorname{MSE}\left(\mathbf{Y}_c \mathbf{M}, \mathbf{Y}_c^* \mathbf{M}\right)
\end{aligned}
$$
$\mathbf{Y}_c$: the mag spectrogram of the target signal

$\mathbf{M}$: the mel-filterbank

Utterance-level permutation invariant training (uPIT) is applied.

# 4. Exp configurations
3 exps:
1. Echoic noisy speech enhancement
2. Echoic noisy speech separation
3. Multichannel noisy ASR

Compared against time-domain beamformers [multi-channel Wiener filter (TD MWF) and TD MVDR], freq-domain beamformers [speech distortion weighted MWF (SDW-MWF) and MVDR], masked-based beamformers [MVDR and GEV beamformers using IBM to estimate the beamforming filters]

# 5. Results and Discussion
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220911204651.png)

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220911204938.png)

# 6. Conclusion
In this paper, proposed FaSNet, a time-domain adaptive beamforming method especially suitable for online, low-latency applications. FaSNet was designed as a two-stage system, where the first stage estimated the beamforming filter for a randomly selected reference mic, and the second stage used the output of the first stage to calculate the filters for all the remaining mics. FaSNet can also be concatenated with any other single-channel system for further performance improvement. Exp results showed that FaSNet achieved better or on par performance than several oracle traditional beamformers on both echoic noisy speech enhancement (ESE) and echoic noisy speech separation (ESS) tasks.