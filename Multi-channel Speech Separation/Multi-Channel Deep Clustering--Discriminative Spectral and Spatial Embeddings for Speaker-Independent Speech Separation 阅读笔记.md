#! https://zhuanlan.zhihu.com/p/579969819
# Multi-Channel Deep Clustering: Discriminative Spectral and Spatial Embeddings 阅读笔记
# Multi-Channel Deep Clustering: Discriminative Spectral and Spatial Embeddings for Speaker-Independent Speech Separation 阅读笔记

# Abstract
Spatial info can be leveraged to differentiate signals from different directions. This study combines spectral and spatial features in a deep clustering framework so that the complementary spectral and spatial info can be simultaneously exploited to improve speech separation. We find that simply encoding inter-micphone phase patterns as additional input features during deep clustering provides a significant improvement in separation performance, even with random microphone array geometry.

# 1. Introduction
Deep clustering addresses the *cocktail party problem* by training a DNN to project each T-F unit to a high-dimensional embedding vector such that the embeddings for the T-F unit pairs dominated by the same speaker are close, while those for pairs dominated by different speakers are farther away from each other. This way, the spk assignment of each T-F unit can be determined at run time by applying a simple clustering algorithm to the embeddings. 

When multiple mics are available, the directional info associated with each source can be exploited for separation, as sound sources are often spatially separated in real-world environments. To utilize this info, conventional wisdom focuses on clustering the individual T-F units into different sources according to their spatial origins by assuming that each T-F unit is dominated by only one source across all the mic channels.

When the sound sources are spatially close, when room reverberation is present, or when the sound sources are moving, the ITDs, ILDs, IPDs and directional statistics are typically not good enough to achieve sufficient source separation. In such cases, spectral info can complement the insufficient spatial info, as sound sources such as speech exhibit characteristic spectral patterns that can be learned, as demonstrated by single-channel DPCL.

Propose to improve DPCL by incorporating spatial info into the input features, along with the usual spectral info, in order to provide a stronger set of separation cues.

Conventional beamforming algorithms is only linear spatial filter per freq. Many factors can reduce the effectiveness of such beamforming: room reverberation, moving sources, diffuse noise, and conditions with more sources than mics, all can significantly degrade the resulting separation. In addition, when sound sources arrive from the same general direction, beamforming may fail to resolve them.

In contrast, by jointly training on spectral and spatial features, our approach can learn to balance the two types of info. The estimated embeddings and masks from the proposed algorithm may also serve as a better initialization for the T-F masking based beamforming approaches. However, in our exps the joint DPCL was often able to outperform the mask-based beamforming approach even when using the ideal oracle masks to obtain the beamforming parameters.

# 2. System description
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202210/20221025225649.png)
## 2.1. Single-Channel Deep Clustering
The key idea of DPCL is to learn a high-dimensional embedding for each T-F unit using a neural network such that the embeddings for the T-F unit pairs dominated by the same speaker are close with farther away otherwise. At test time, the speaker assignment of each T-F unit can then be simply determined using a clustering algorithm, such as K-means, on the learned embeddings. 

## 2.2. Two-Channel Deep Clustering
Here we encode not only spectral but also spatial info into the embedding of each T-F unit by including spatial features as additional inputs, as illustrated in Fig. 1(b).

The narrowband approach performs clustering within each freq band using spatial cues such as IPDs or ILDs. DUET algorithm assumes that the mic pairs are placed sufficiently close to each other so that phase-wrapping effects can be neglected. It estimates ITD of each T-F unit by directly dividing the phase difference by the angular freq, and then performs clustering on the estimated ITDs and ILDs of all the T-F units. Unfortunately, with narrowly separated mics, ITDs could be too small to be useful for separation. Moreover room reverberation can substantially deteriorate ITDs and ILDs.

The cosIPD and sinIPD features at diff freq bands are very diff, so we combine them with spectral features that can help resolve the ambiguity.

The wideband approach avoids the IPD ambiguity by enumerating a set of potential time delays. The key insight is that, given a time delay, the phase differences at all the frequency bands can be unambiguously determined in anechoic environments.
MESSL [Model-Based Expectation-Maximization Source Separation and Localization] algorithm therefore performs spatial clustering according to the time delays by checking whether the hypothesized time delay matches the observed phase differences at different frequencies. Motivated by MESSL and GCC-PHAT, we derive the following spatial feature for model training:
$$
\operatorname{GCC}(t, f, p, q, \tau)=\cos \left(\theta_{t, f, p, q}-\frac{2 \pi f}{N} \tau\right)
$$
$N$: the frame length

$\theta_{t, f, p, q}$: the observed phase difference between STFT coefficients at time $t$ and freq $f$ of signals at mics $p$ and $q$.

$\tau$: the hypothesized time delay in samples

$\frac{2 \pi f}{N} \tau$: the hypothesized phase difference. The $2\pi$-periodic cosine operation here can deal with potential phase wrapping effects.

The rationale behind this feature is that each of the underlying sources in a mixture could come from any direction. Our approach avoids a separate sound localization module by enumerating a set of potential time delays. When a hypothesized time delay matches the observed phase diff, it appears as a peak in the derived spatial feature. Although this feature exhibits strong spatial aliasing effects in high-freq bands, we hand it over to a neural network which may learn to deal with the spatial aliasing effects automatically.

GCC features have a much higher dimension than the spectral features. If each dimension of these two features is normalized to unit variance, more importance will be implicitly placed on the GCC features. However, spectral features are also very important for DPCL. Our system places equal importance on them by normalizing each dimension of the spatial features to have ${1}/{K}$ variance, where $K$ is the number of the time delays of interest, and each dimension of the spectral features to have unit variance. This simple strategy leads to faster convergence and better performance compared with normalizing all the dimensions to unit variance in our exps.

## 2.3. Multi-Channel Deep Clustering
First choose a ref mic, and for each mic pair, get an embedding for each T-F unit using the two-channel deep clustering model. Then stack the embeddings of all the pairs and perform k-means clustering on the stacked embeddings. The resulting binary mask is applied to the ref mic signal for separation. This way, our algorithm is readily applicable to mic arrays with diverse mic geometries and with any number of mics.

# 4. Evaluation Results
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221102162955.png)
Reverberation blurs the spectral features and breaks the assumption of sparsity in the speech spectrogram, making mask-based separation more difficult. 8.8 dB indicates GCC's effectiveness for encoding spatial info.

cosIPD feature matches a single dimension of GCC: $\operatorname{cosIPD}(t, f, p, q) = \operatorname{GCC}(t, f, p, q, 0)$

# 5. Conclusion
Proposed a novel approach to combine deep clustering with spatial clustering for blind source separation. By including phase difference features in the input to a deep clustering network, we can encode both spatial and spectral info in the embeddings it creates, leading to better estimated T-F masks. 