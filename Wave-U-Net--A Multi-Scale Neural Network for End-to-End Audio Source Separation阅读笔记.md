#! https://zhuanlan.zhihu.com/p/511508780
# Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation阅读笔记

# Abstract & Conclusion

Models for audio source separation usually operate on the magnitude spectrum, which ignores phase information and makes separation performance dependant on hyper-parameters for the spectral front-end. Therefore, we investigate end-to-end source separation in the time-domain, which allows modelling phase information and avoids fixed spectral transformations. Due to high sampling rates for audio, employing a long temporal input context on the sample level is difficult, but required for high quality separation results because of long-range temporal correlations.

Propose the Wave-U-Net, an adaptation of the U-Net to the one-dimensional time domain, which repeatedly resamples feature maps to compute and combine features at different time scales.

Proposed the Wave-U-Net for end-to-end audio source separation without any pre- or postprocessing. A long temporal context is processed by repeated downsampling and convolution of feature maps to combine high- and low-level features at different time-scales. It outperforms the SOTA spectrogram-based U-Net architecture when trained under comparable settings.

We highlight the lack of a proper temporal input context in recent separation and enhancement models, which can hurt performance and create artifacts, and propose a simple change to the padding of convolutions as a solution.

Artifacts resulting from upsampling by zero-padding as part of strided transposed convolutions can be addressed with a linear upsampling with a fixed or learned weight to avoid high-freq artifacts.

# Introduction

Spectrogram based approach has limitations.

1. STFT depends on many params, such as the size and overlap of audio frames, which can affect the time and frequency resolution. In practice, the transform params are fixed to specific values.
2. The separation model does not estimate the source phase, it is assumed to be equal to the mixture phase, which is incorrect for overlapping partials.

It would be desirable for the separation model to learn to estimate the source signals including their phase directly.

Hard to deal effectively with the very long-range temporal dependencies present in audio due to its high sampling rate.

Contributions:

1. Propose Wave-U-Net, which separates sources directly in the time domain and can take large temporal context into account.
2. Replace strided transposed convolution for upsampling feature maps with linear interpolation followed by a normal convolution to avoid artifacts.

![image-20220506185036547](https://tva1.sinaimg.cn/large/e6c9d24ely1h1yvowxjfkj20kf0lxjtr.jpg)

# Related Work

TasNet performs a decomposition of the signal into a set of basis signals and weights, then creates a mask over the weights which are finally used to reconstruct the source signals.

MRCAE (multi-resolution convolutional auto-encoder) uses two layers of convolution and transposed convolution each.

# The Wave-U-Net Model

mixture waveform $M\in[-1,1]^{L_m\times C}$

$K$ source waveforms $S^k\in[-1,1]^{L_s\times C}$

$C$ is number of audio channels

For model variants with extra input context, we have $L_m\gt L_s$ and make predictions for the centre part of the input.

## 3.1 The base architecture

It computes an increasing number of higher-level features on coarser time scales using downsampling (DS) blocks. These features are combined with the earlier computed local, high-resolution features using upsampling (US) blocks, yielding multi-scale features which are used for making predictions. Each level in the network operates at half the time resolution as the previous one.

![image-20220506225142082](https://tva1.sinaimg.cn/large/e6c9d24ely1h1z2nrckwnj20jw0da0uv.jpg)

Conv1D(x,y) denotes a 1D convolution with $x$ filters of size $y$.

Decimate discards features for every other time step to halve the time resolution.

Upsample performs upsampling in the time direction by a factor of two, for which we use linear interpolation.

Concat(x) concatenates the current, high-level features with more local features x.

### 3.1.1 Avoiding aliasing artifacts due to upsampling

Using transposed convolutions with strides to upsample feature maps can introduce aliasing effects (混叠效应) in the output.

[Aliasing is the effect of new frequencies appearing in the sampled signal after reconstruction, that were not present in the original signal.]

![image-20220507110409457](https://tva1.sinaimg.cn/large/e6c9d24ely1h1znttvh4vj20k40mtq6g.jpg)

### 3.2.1 Different output layer

Baseline model outputs one source estimate for each of $K$ sources by independently applying $K$ convolutional filters followed by a tanh non-linearity to the last feature map.

We apply only $K-1$ convolutional filters with a size of 1 to the last feature map of the network, followed by tanh non-linearity, to estimate the first $K-1$ source signals. The last source is simply computed as $\hat{S}^K=M-\sum_{j=1}^{K-1}\hat{S}^j$.

Similar idea can be found in mask-based speech enhancement and speech separation.

### 3.2.2 Prediction with proper input context and resampling

The temporal context for a excerpt of audio is given in the full audio signal but is ignored and assumed to be silent by zero padding.

Without proper context information, the network thus has difficulty predicting output values near the beginning and end of the sequence. As a result, simply concatenating the outputs as non-overlapping segments at test time to obtain the prediction for a full audio signal can create audible artifacts at the segment borders, as neighbouring outputs can be inconsistent when they are generated without correct context information. 

As a solution, we employ convolutions without implicit padding and instead provide a mixture input larger than the size of the output prediction, so that the convolutions are computed on the correct audio context.

### 3.2.4 Learned upsampling for Wave-U-Net

Linear interpolation for upsampling is simple, parameter-less and encourages feature continuity. However, it may be restricting the network capacity too much as the feature spaces used in these feature maps may be not structured.

A learned upsampling could further enhance performance. We thus propose the learned upsampling layer.

For a given $F\times n$ feature map with $n$ time steps, we compute an interpolated feature $f_{t+0.5}\in\mathbb{R}^F$ for pairs of neighbouring features $f_t,f_{t+1}\in\mathbb{R}^F$ using parameters $w\in\mathbb{R}^F$ and the sigmoid function $\sigma$ to constrain each $w_i\in w$ to the [0,1] interval:
$$
f_{t+0.5}=\sigma(w) \odot f_{t}+(1-\sigma(w)) \odot f_{t+1}
$$
This can be implemented as a 1D convolution across time with $F$ filters of size two and no padding with a properly constrained matrix.

# Experiments

MSE over all source output samples in a batch.

$L_m=L_s=16384$ input and output samples, $L=12$ layers, $F_c = 24$ extra filters per layer