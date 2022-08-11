#! https://zhuanlan.zhihu.com/p/500993159
# Multi-task Neural Network for Robust Multiple Speaker Embedding Extraction 阅读笔记

# Abstract & Conclusion

A multi-task NN-based approach for extracting speaker embeddings from audio mixtures of multiple overlapping voices.

The network first extracts a latent feature for each direction. This feature is used for detecting sound sources as well as identifying speakers. The proposed method does not rely on explicit sound source separation. The NN model learns from data to extract the most suitable features of the sounds at different directions.

Presented a novel multi-task neural network for extracting speaker embeddings of multiple simultaneous speakers using DOA estimation as an auxiliary task. The NN learns to estimate a spatial spectrum score and a speaker embedding for each direction. The spatial spectrum is used as weighting parameters for weighted average and standard deviation pooling of the frame-wise speaker features along the time axis.

# Intro

Knowing speaker identities allows natural long-term interactions, as well as speaker-adapted ASR.

In an ideal embedding space, distances between voices of the same speaker are smaller than distances between voices of different speakers.

Joint DOA estimation and recognition of multiple speakers has not been studied so far.

This paper investigates DNNs for speaker recognition under the multi-speaker condition using DOA estimation as an auxiliary task. We use the NNs to extract features for each direction, which are shared for both DOA estimation and speaker embedding. 

Our proposed NN shares similarities with the well-known x-vector network, that both networks are trained using speaker identification loss and extract speaker embeddings from hidden layers.



# Approach

## 2.1. Network Input

Raw STFT, which includes both the spectral power info as well as the phase info of the input signal.

For DOA estimation, *Inter-channel Level Difference* (ILD) and spectral cues can be extracted from the power info, and *Inter-channel Phase Difference* (IPD) can be extracted from the phase info. 

The input feature of each unit has a dimension of $T\times337\times8$, where the number of frames $T$ varies across different segments.

## 2.2. Network Output and Loss Function

Network output includes frame-wise prediction of the spatial spectrum $\bold{p}_t=\{p_{td}\}_{d=1}^D\in[0,1]^D$ for DOA estimation, and segment-wise prediction of speaker posterior probability at each direction $\bold{q}_d=\{q_{ds}\}_{s=1}^S\in[0,1]^D$ for speaker identification.

The subscripts $t$ is the frame index, $d\in\{1,2,...,D\}$ is the direction index, and $s\in\{1,2,...,S\}$ is the speaker ID.

**Encoding.** The desired output spatial spectrum is encoded by the Gaussian-based spatial spectrum coding.

The speaker ID prediction at direction $\varphi_d$ depends on the nearest sound source (speaker) to that direction, that is:
$$
q_{ds}=\left\{\begin{array}{l}
1\ \ \ \ \text{if Speaker }s\text{ is the nearest speaker to }\varphi_d \\
0\ \ \ \ \text{otherwise} \\
\end{array}\right.
$$
**Loss Functions.** The target loss func is a linear combination of the individual task-specific loss functions:
$$
\text{Loss}=\mu\text{Loss}_{DOA}+\lambda\text{Loss}_{ID}
$$
where $\mu$ and $\lambda$ are weighting parameters. We use the MSE loss for DOA estimation. The speaker identification loss is the weighted sum of cross entropy loss a individual directions:
$$
\text{Loss}_{ID}=-\sum_{d=1}^{D}w_d\sum_{s=1}^{S}q_{ds}log\hat{q}_{ds}
$$
The weighting {$w_d$} depends on its distance to the DOAs of the sound sources, following the Gaussian distribution as the spatial spectrum. The only difference is that $\sigma$ is set to 16 instead of 8.

**Decoding.** The network outputs frame-wise spatial spectra $\bold{p}_t$ and speaker embedding $\bold{r}_d$ per direction. We compute the avg of the frame-wise spatial spectra to get segment-level DOA prediction:
$$
\bold{p}=\frac{1}{T_0}\sum_{t=1}^{T_0}\bold{p}_t
$$
and apply peak finding to detect sound sources. For any detected sound source, the speaker embedding output at the estimated direction is the predicted speaker embedding.

## 2.3. Network Architecture

Design a multi-task network for speaker embedding using DOA estimation as an auxiliary task.

![image-20220415200628427](https://tva1.sinaimg.cn/large/e6c9d24ely1h1anvdj5k8j20sk1dqagc.jpg)

The green blocks applies 2D convolutions along time and freq axes to extract TF local features.

The red blocks starts with two layers of 2D conv to extract frame-wise speaker features per direction $\bold{f}_{td}\in\mathbb{R}^{512}$, which is then pooled along the time axis using *weighted average and standard deviation*:
$$
\begin{aligned}
\mathbf{f}_{d}^{(a v g)} &=\frac{\sum_{t=1}^{T_{o}} p_{t d} \mathbf{f}_{t d}}{\sum_{t=1}^{T_{o}} p_{t d}} \\
\mathbf{f}_{d}^{(s t d)} &=\sqrt{\frac{\sum_{t=1}^{T_{o}} p_{t d}\left(\mathbf{f}_{t d}-\mathbf{f}_{d}^{(a v g)}\right)^{2}}{\sum_{t=1}^{T_{o}} p_{t d}}}
\end{aligned}
$$
We use the output of the DOA estimation branch {$p_{td}$} as the weighting parameters, because the DOA estimation output (i.e. spatial spectrum) indicates whether there is an active sound at that frame and direction. Their concatenation $\bold{f}_d=[\mathbf{f}_{d}^{(a v g)}\mathbf{f}_{d}^{(s t d)}]\in\mathbb{R}^{1024}$ is the segment-level speaker feature per direction. Then the speaker identity posterior probability is computed from these features with FC layers ($1\times1$ convolutions) and a softmax layer.

At test time, the 512-dimensional activation of the last hidden layer after BN is the speaker embedding $\bold{r}_d$ at the direction $\varphi_d$.



# Experiments

SSLR + simulated VoxCeleb1

