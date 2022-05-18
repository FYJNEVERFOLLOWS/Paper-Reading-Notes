# [多通道语音增强] Improved Speech Enhancement with the Wave-U-Net 阅读笔记

# Abstract & Conclusion

Wave-U-Net is an end-to-end learning method for audio source separation operates directly in the time domain, permitting the integrated modelling of phase information and being able to take large temporal contexts into account. We find that a reduced number of hidden layers is sufficient for speech enhancement in comparison to the original system designed for singing voice separation in music.

It is possible that the advantages stems from the upsampling that avoids aliasing, which should be further investigated.

The results indicate that there is room for increasing effectiveness and efficiency by further adapting the model size and other parameters, e.g. filter sizes, to the task and expanding to multi-channel audio and multi-source-separation.

# Introduction

Audio source separation refers to the problem of extracting one or more target sources while suppressing interfering sources and noise. Two related tasks are those of speech enhancement and singing voice separation, both of which involve extracting the human voice as a target source.

# Related work

Time domain model: Wavenet, SEGAN

Wavenet has a non-causal conditional input and a parallel output of samples for each prediction and is based on the repeated application of dilated convolutions with exponentially increasing dilation factors to factor in context information.

SEGAN employs a neural network in the time-domain with an encoder and decoder pathway that successively halves and doubles the resolution of feature maps in each layer, respectively, and features skip connections between encoder and decoder layers.

# Wave-U-Net for Speech Enhancement

The overall architecture is a one-dimensional U-Net with down and upsampling blocks.

Wave-U-Net uses a series of downsampling and upsampling blocks to make its predictions.

In applying the Wave-U-Net architecture to the application of speech enhancement, our objective is to separate a mixture waveform $m\in[-1,1]^{L\times C}$ into $K$ source waveforms $S^1,...S^k$ with $S^k\in[-1,1]^{L\times C}$ for all $k\in1,...,K$.

$C$: the number of audio channels

$L$: the number of audio samples

$K=2$ and $C=1$ in our case of monaural speech enhancement.

$K\cdot C$ filters is applied to convert the stack of features at each audio sample into a source prediction for each sample.

# Experiments

![image-20220513200613902](https://tva1.sinaimg.cn/large/e6c9d24ely1h2717pmabsj20vq09575m.jpg)

![image-20220513200631677](https://tva1.sinaimg.cn/large/e6c9d24ely1h27187287sj20vg092myl.jpg)
