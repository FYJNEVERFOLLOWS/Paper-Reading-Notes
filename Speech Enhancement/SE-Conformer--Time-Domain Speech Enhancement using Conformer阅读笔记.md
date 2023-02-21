#! https://zhuanlan.zhihu.com/p/506291977
# [时域语音增强] SE-Conformer: Time-Domain Speech Enhancement using Conformer阅读笔记

# Abstract & Conclusion

Conformer can capture both the short and long-term temporal sequence information by attending to the whole sequence at once with multi-head self-attention and convolutional neural network. Propose an end-to-end SE-Conformer, incorporating a convolutional encoder-decoder and conformer, designed to be directly applied to the time-domain signal. It is suitable for sequence modeling by attending the entire sequence at once with self-attention and CNN in latent space.

# Intro

One of the mainstream applications of DL for SE is algorithms based on the TF domain, which is computed with a STFT. This can be further divided into two approaches: mask-based target and mapping-based target. Mask-based approaches estimate the ideal ratio mask from noisy acoustic features, then multiply noisy magnitude spectra, and reconstruct clean speech signals. Mapping-based targets estimate the clean magnitude spectrum from a noisy magnitude. These approaches cause audible artifacts only by estimating the magnitude spectrum of the waveform while maintaining noisy phase information.

Another stream of SE research is the time-domain end-to-end method, which is designed to directly estimate clean speech waveforms from noisy speech waveforms.

ConvTasnet: CED + TCN

Demucs: CED + BLSTM

CED: convolutional encoder–decoder with skip connection structure

The conformer block is more appropriate for Convolutional Encoder-Decoder-based structures because of its enhanced ability to reflect the local and global temporal context dependencies by attending the entire sequence in the latent representation.

![image-20220425233039424](https://tva1.sinaimg.cn/large/e6c9d24ely1h1mdyxrze2j21730u0tcq.jpg)

# Method

The proposed model consists of a multi-layer CED structure and conformer block. The convolutional encoders perform upsampling and convolution blocks sequentially on the waveform signal to obtain the corresponding latent representations. These representations are applied with conformer block that capture the local context and global context dependencies to model the sequence information. The decoder performs the convolution blocks with downsampling to reconstruct the time-domain signal from a latent representation estimated through the conformer block.

For training objectives, we used L1 and multi-resolution STFT loss as follows:
$$
L_{t o t a l}(\mathbf{x}, \hat{\mathbf{x}})=\frac{1}{T}\left[\|\mathbf{x}-\hat{\mathbf{x}}\|_{1}+\sum_{m=1}^{M} L_{s t f t}^{(m)}(\mathbf{x}, \hat{\mathbf{x}})\right]
$$
where the multi-resolution STFT loss is the sum of the STFT losses, which is the sum of the spectral convergence (sc) and magnitude (mag) loss, represented as follows:
$$
\begin{gathered}
L_{s t f t}^{(m)}(\mathbf{x}, \hat{\mathbf{x}})=L_{s c}^{(m)}(\mathbf{x}, \hat{\mathbf{x}})+L_{m a g}^{(m)}(\mathbf{x}, \hat{\mathbf{x}}) \\
L_{s c}^{(m)}(\mathbf{x}, \hat{\mathbf{x}})=\frac{\left\|S T F T^{(m)}(\mathbf{x})|-| S T F T^{(m)}(\hat{\mathbf{x}})\right\|_{F}}{\|S T F T(\mathbf{x})\|_{F}} \\
L_{m a g}^{(m)}(\mathbf{x}, \hat{\mathbf{x}})=\frac{1}{T}\left\|\log \left|S T F T^{(m)}(\mathbf{x})\right|-\log \mid S T F T^{(m)}(\hat{\mathbf{x}})\right\|_{1}
\end{gathered}
$$
where $||\cdot||_F$ and $||\cdot||_1$ denote the Frobenius and L1 norms. $M$ and $|STFT^{(m)}(\cdot)|$ denote the number of resolution parameter sets for STFT and the magnitude of STFT with the $m$th analysis parameter set.

[Note]: In the STFT-based time-freq representation of signals, there is a trade-off between time and frequency resolution; e.g., increasing window size gives higher frequency resolution while reducing temporal resolution. By combining multiple STFT losses with different analysis parameters (i.e., FFT size, window size, and frame shift), it greatly helps the generator to learn the time-freq characteristics of speech. Moreover, it also prevents the generator from being overfit to a fixed STFT representation, which may result in suboptimal performance in the waveform-domain.

## 2.1. Encoder

The encoder takes the mixture noise-corrupted waveform as an input $\bold{x}\in\mathbb{R}^T$ and learns a latent representation using the $R$ upsampling blocks and the $L$ stack of convolutional blocks (E-ConvBlock). The upsampling block doubled the time resolution using sinc interpolation.

## 2.2. Conformer-based sequence modeling in latent space

The conformer can model local context information by inserting a depth-wise convolution into a transformer, which is effective in global context information modeling.

The conformer is the $N$ stack of a conformer block shown in Fig.1(b).

## 2.3. Decoder

The decoder takes the output of the conformer blocks and sequentially performs $L$ convolutional blocks (D-ConvBlock) and $R$ downsample blocks to estimate the clean waveform $\hat{\bold{x}}\in\mathbb{R}^T$. The downsampling blocks halved the time resolution by maintaining the number of features.

# Experiments

Clean speech from VCTK, nosie types from DEMAND

50 hrs of clean speech samples from Librispeech

sr: 16kHz

CSIG: a signal distortion mean opinion score

CBAK measures background intrusiveness

COVL measures speech quality

PESQ: perceptual evaluation of speech quality

STOI: short-time objective intelligibility measures the intelligibility gain by processing the noisy mixture with reference to the clean

Remix, and BandMask augmentation on-the-fly during the training of the models



### 3.4.1. Comparison with previous methods

![image-20220427141521078](https://tva1.sinaimg.cn/large/e6c9d24ely1h1o95q8v6jj22gc0rg471.jpg)

The results indicated that the proposed method preserves better speech quality.

To demonstrate the generalization of the proposed model's improvements over baselines, we experimented on a larger Librispeech dataset, as shown in Table 2.

![image-20220427142151181](https://tva1.sinaimg.cn/large/e6c9d24ely1h1o9cg1lnsj21si0u0tgu.jpg)

We found that the proposed model tended to perform better than the baselines in all noisy environments. We can confirm that the proposed model could achieve better speech quality in various background noise for the simulated large data.

### 3.4.2. BLSTM/Transformer/Conformer Block

![image-20220427142609983](https://tva1.sinaimg.cn/large/e6c9d24ely1h1o9gy1euqj21660f40vt.jpg)

### 3.4.3. The effect of the components of Conformer

![image-20220427143136677](https://tva1.sinaimg.cn/large/e6c9d24ely1h1o9mlnzs7j21620jeq67.jpg)

Notably, we can see a significant drop in performance when ConvBlock is removed from the conformer block. This can be interpreted as ConvBlock being an important factor in capturing local context information based on the transformer model.

