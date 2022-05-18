#! https://zhuanlan.zhihu.com/p/511508467
# Real Time Speech Enhancement in the Waveform Domain阅读笔记

# Abstract & Conclusion

Present a causal speech enhancement model working on the raw waveform that runs in real-time on a laptop CPU. It is based on an encoder-decoder architecture with skip-connections and optimized on both time and frequency domains, using multiple loss functions.

Suggest data augmentation techniques (reverb with two sources, partial dereverberation) applied directly on the raw waveform which further improve model performance and its generalization abilities.

Turn DEMUCS /dima:cs/ into a causal speech enhancer, processing audio in real time on consumer level CPU.

# Introduction

For many applications like audio and video calls, hearing aids and ASR, a key feature of a speech enhancement system is to run in real time and with as little lag as possible (online).

Traditional methods have trouble dealing with common noises such as non-stationary noise or a babble noise which is encountered when a crowd of people are simultaneously talking. The presence of such noise types degrades hearing intelligibility of human speech greatly.

Proposed a real-time version of the DEMUCS architecture adapted for speech enhancement. It consists of a causal model, based on convolutions and LSTMs, with a frame size of 40ms, a stride of 16ms, and that runs faster than real-time on a single laptop CPU core. For audio quality purposes, our model goes from waveform to waveform, through hierarchical generation (using U-Net like skip-connections). We optimize the model to directly output the "clean" version of the speech signal while minimizing a regression loss function (L1 loss), complemented with a spectrogram domain loss. Data augmentation techniques: frequency band masking and signal reverberation.

Multiple metrics to measure speech enhancement systems have shown to not correlate well with human judgements. Hence, we report results for both objective metrics as well as human evaluation. Conduct an ablation study over the loss and augmentation functions to better highlight the contribution of each part. Finally, we analyzed the artifacts of the enhancement process using WER produced by an ASR model.

# Model

## 2.1. Notations and problem settings

Focus on monaural (single-mic) speech enhancement that can operate in real-time applications.

Given an audio signal $x\in\mathbb{R}^T$, composed of a clean speech $y\in\mathbb{R}^T$ that is corrupted by an additive background signal $n\in\mathbb{R}^T$ so that $x=y+n$. The length, $T$, is not a fixed value across samples, since the input utterances can have different durations.

## 2.2. DEMUCS architecture

DEMUCS consists in a multi-layer convolutional encoder and decoder with U-net skip connections, and a sequence modeling network applied on the encoders' output. It is characterized by its number of layers $L$, initial number of hidden channels $H$, layer kernel size $K$ and stride $S$ and resampling factor $U$. The encoder and decoder layers are numbered from 1 to $L$ (in reverse order for the decoder).

Formally, the encoder network $E$ gets as input the raw wavefrom and outputs a latent representation $E(x)=z$. Each layer consists of a conv layer with a kernel size of $K$ and stride of $S$ with $2^{i-1}H$ output channels, followed by a ReLU activation, a "$1\times1$" convolution with $2^iH$ output channels and finally a GLU activation that converts back the number of channels to $2^{i-1}H$, see Fig 1b for a visual description.

![image-20220508205052533](https://tva1.sinaimg.cn/large/e6c9d24ely1h21aenwatxj218i0ion1h.jpg)

Next, a sequence modeling $R$ network takes the latent representation $z$ as input and outputs a non-linear transformation of the same size, $R(z)=LSTM(z)+z$, denoted as $\hat{z}$. The LSTM network consists of 2-layers and $2^{L-1}H$ hidden units. For causal prediction, we use an unidirectional LSTM, while for non causal models, we use a bidirectional LSTM, followed by a linear layer to merge the both outputs.

Lastly, a decoder network $D$, takes as input $\hat{z}$ and outputs an estimation of clean signal $D(\hat{z})=\hat{y}$.

We noticed that upsampling the audio by a factor $U$ before feeding it to the encoder improves accuracy. We downsample the output of the model by the same amount. The resampling is done using a sinc interpolation filter, as part of the end-to-end training, rather than a pre-processing step.

## 2.3. Objective

We use the L1 loss over the waveform together with a multi-resolution STFT loss over the spectrogram magnitudes proposed by [Parallel wavegan: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram, Probability density distillation with generative adversarial networks for high-quality parallel waveform generation].

We define the STFT loss to be the sum of the *spectral convergence (sc)* loss and the *magnitude* loss as follows,
$$
\begin{aligned}
L_{\mathrm{sttt}}(\boldsymbol{y}, \hat{\boldsymbol{y}}) &=L_{s c}(\boldsymbol{y}, \hat{\boldsymbol{y}})+L_{m a g}(\boldsymbol{y}, \hat{\boldsymbol{y}}) \\
L_{s c}(\boldsymbol{y}, \hat{\boldsymbol{y}}) &=\frac{\||S T F T(\boldsymbol{y})|-|S T F T(\hat{\boldsymbol{y}})|\|_{F}}{\||\operatorname{STFT}(\boldsymbol{y})|\|_{F}} \\
L_{m a g}(\boldsymbol{y}, \hat{\boldsymbol{y}}) &=\frac{1}{T}\|\log |\operatorname{STFT}(\boldsymbol{y})|-\log |\operatorname{STFT}(\hat{\boldsymbol{y}})|\|_{1}
\end{aligned}
$$
$y$: the clean signal 

$\hat{y}$: the enhanced signal 

$F$: Frobenius norm

Total loss is:
$$
L_{total}(y,\hat{y})=\frac{1}{T}\left[\|\boldsymbol{y}-\hat{\boldsymbol{y}}\|_{1}+\sum_{i=1}^{M} L_{\mathrm{stft}}^{(i)}(\boldsymbol{y}, \hat{\boldsymbol{y}})\right]
$$
where $M$ is the number of STFT losses, multi-resolution: number of FFT bins $\in \{512,1024,2048\}$, hop sizes $\in \{50, 120, 240\}$, window lengths $\in\{240,600,1200\}$.

# Experiments

**Evaluation Methods** 

Objective measures: PESQ [-0.5, 4.5], STOI [0, 100], CSIG [1, 5], CBAK [1, 5]; COVL [1, 5]

Subjective measure: MOS [CrowdMOS package]

![image-20220508233155862](https://tva1.sinaimg.cn/large/e6c9d24ely1h21f26waxij215a0hqtcu.jpg)

**Data augmentation**

Apply a random shift between 0 and $S$ seconds. 

The *Remix* augmentation shuffles the noises within one batch to form new noisy mixtures.

*BandMask* is a band-stop filter with a stop band between $f_0$ and $f_1$, sampled to remove 20% of the frequencies uniformly in the mel scale. This is equivalent, in the waveform domain, to the SpecAug augmentation used for ASR training.

*Revecho*: given an initial gain $\lambda$, early delay $\tau$ and RT60, it adds to the noisy signal a series of $N$ decaying echos of the clean speech and noise. $\lambda,\tau$ and RT60 are sampled uniformly respectively over [0, 0.3], [10, 30] ms, [0.3, 1.3] sec.

# Related Work

Traditional speech enhancement methods generate either an enhanced version of the magnitude spectrum (mapping) or produce an estimate of the ideal binary mask that is then used to enhance the magnitude spectrum (masking).

other papers worth reading...

Multi-objective learning and mask-based post-processing for deep neural network based speech enhancement

SEGAN



