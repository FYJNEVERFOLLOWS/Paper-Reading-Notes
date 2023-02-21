# SpEx: Multi-Scale Time Domain Speaker Extraction Network 阅读笔记

# 1. Introduction
Contributions:
1) Emulate human's ability of selective auditory attention by mimicking the top-down voluntary focus using a speaker encoder.
2) Propose a time-domain solution as an extension to Conv-TasNet from speech separation to speaker extraction, that avoids the phase estimation in freq-domain approaches.
3) Propose a multi-task learning algorithm to jointly optimize the four network components of SpEx with an unified training process.
4) Propose a multi-scale encoding and decoding scheme that captures multiple temporal resolutions for improved voice quality.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202210/20221005165619.png)

# 2. Time-domain Speaker Extraction Network
Similar to TasNet, we opt for a trainable neural network to serve as the speech encoder in time-domain speaker extraction. The speaker encoder is trained to convert time-domain speech signal into spectrum-like embedding, also called embedding coefficients.

To benefit from the idea of speaker encoder and task-oriented optimization [Single channel target speaker extraction and recognition with speakerbeam] [Optimization of speaker extraction neural network with magnitude and temporal spectrum approximation loss] [Compact network for speakerbeam target speaker extraction], propose a multi-task learning algorithm to incorporate the speaker encoder as part of the SpEx network.

## A. SpEx Architecture
### 2) Speech Encoder
We can consider the filters in the convolutional layers as the basis functions in analogy to the sines and cosines in the freq domain. Time domain encoding is different from Fourier transform in that a) the feature representations don't handle the real & imag parts separately; b) the basis functions are not pre-defined as sines or cosines, but rather trainable from the data.

This paper proposes to encode the mixture speech into multi-scale speech embeddings using several parallel 1-D CNNs with $N$ filters each for various temporal resolutions. The filters of the parallel 1-D CNNs are of different lengths, $L_1(short)$, $L_2(middle)$, $L_3(long)$ samples, to cover different window sizes.

To concatenate the embeddings across different time-scale, we align them by keeping the same stride, $L_1/2$, across different scales. The encoder learns representations in multiple scales with the varying filter lengths, e.g., the short window has good resolution at high freq and long window has high resolution at low freq. Without trading the temporal resolution with freq resolution like in STFT, we encode the time-domain signal into three temporal resolutions in the embedding $E$.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202210/20221007161452.png)

## B. Multi-scale Encoding and Decoding
Speech has a rich temporal structure over multiple time scales presenting phonemic, prosodic and linguistic content. [Testing multi-scale processing in the auditory system] showed that speech analysis of multiple temporal resolutions leads to improved speech recognition performance.

Multi-scale embedding coefficients $E=[E_1E_2E_3]$

Multi-scale masks $M_1,M_2,M_3$

Multi-scale modulated responses $S_1,S_2,S_3$

Multi-scale SI-SDR loss:
$$
J_1=-[(1-\alpha-\beta)\rho(s_1,s)+\alpha\rho(s_2,s)+\beta\rho(s_3,s)]
$$

SI-SDR loss:
$$
\rho(\hat{s}, s)=10 \log _{10}\left(\frac{\left\|\frac{\langle\hat{s}, s\rangle}{\langle s, s\rangle} s\right\|^2}{\left\|\frac{\langle\hat{s}, s\rangle}{\langle s, s\rangle} s-\hat{s}\right\|^2}\right)
$$

