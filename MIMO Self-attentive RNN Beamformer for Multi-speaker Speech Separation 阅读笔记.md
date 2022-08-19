# MIMO Self-attentive RNN Beamformer for Multi-speaker Speech Separation 阅读笔记

# Abstract
Our proposed ADL-MVDR BF outperformed conventional MVDR by replacing the matrix inversion and eigenvalue decomposition with 2 RNNs. Present a self-attentive RNN BF by leveraging on the powerful modeling capability of self-attention. Temporal-spatial self-attention module is proposed to better learn the beamforming weights from the speech and noise spatial covariance matrices. The temporal self-attention module could help RNN to learn global statistics of covariance matrices. The spatial self-attention module is designed to attend on the cross-channel correlation in the covariance matrices. Furthermore, a multi-channel input with multi-speaker directional features and multi-speaker speech separation outputs (MIMO) model is developed to improve the inference efficiency.

# Introduction
Same motivation as GRNN-BF

3 main contributions:
1. Propose a temporal-spatial self-attention module to learn the beamforming weights. The spatial self-attention can learn the cross-channel correlations from the covariance matrices. The temporal self-attention is designed to consolidate the RNNs to capture the long-term statistics of covariance matrices.
2. The temporal self-attention, the spatial self-attention, and the RNN are complementary to each other.
3. Unlike our previous target speech separation, this model is a MIMO model to enable the inference computation efficiency. It means that multi-speaker speech separation outputs could be obtained simultaneously by feeding with multiple speaker-dependent directional features.

# MIMO self-attentive RNN beamformer
## 2.1. Problem definition
Noisy speech mixture $y=[y_1,y_2,...,y_M]$ in the STFT domain:
$$
\begin{aligned}
\mathbf{Y}(t, f) &=\sum_{i=1}^{C} \mathbf{S}_{i}(t, f)+\mathbf{U}(t, f) \\
\mathbf{N}_{i}(t, f) &=\sum_{j \neq i}^{C} \mathbf{S}_{j}(t, f)+\mathbf{U}(t, f)
\end{aligned}
$$
where $\mathbf{S}_{i}$ and $\mathbf{U}$ represent the $i$-th speaker's reverberated speech and background noise. $\mathbf{N}_i$ is the corresponding interfering noise (the sum of other speakers' interfering speech and the background noise) of the $i$-th speaker. $C$ is the total number of overlapped speakers in the utterance.

Our task aims at separating all of the speakers' speech $\mathbf{S}_{i}$ simultaneously. The input includes LPS, IPD and multiple speakers' DOA $\theta$. The speaker-dependent directional feature (DF($\theta$)) could be calculated based on the DOAs. With the speaker-dependent DOA info, the model could easily figure out the order of the separation outputs and avoid the speaker permutation problem.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220816154528.png)
As shown in Fig. 1, the whole system consists of two parts, a complex-valued ratio filter(cRF) estimator and the proposed self-attentive RNN beamformer.

## 2.2. Generalized RNN-beamformer baseline
GRNN-BF used the cRF to calculate the target speech and noise covariance matrices. The cRF is just an extended version of complex-valued ratio mask (cRM) by using the neighboring context info. Then a RNN was applied to learn the frame-level beamforming weights directly from the covariance matrices. As shown in Fig. 1, the cRF estimator first predicts the target speech and noise cRFs. Then the estimated $i$-th speaker's speech $\hat{\mathbf{S}}_{i}(t, f)$ is
$$
\hat{\mathbf{S}}_{i}(t, f)=\sum_{\tau_{1}=-L}^{\tau_{1}=L} \sum_{\tau_{2}=-L}^{\tau_{2}=L} \mathrm{cRF}_{\mathbf{S}_{i}}\left(t+\tau_{1}, f+\tau_{2}\right) * \mathbf{Y}\left(t+\tau_{1}, f+\tau_{2}\right)
\tag{3}
$$
where $L$ defines the neighboring context size across the freq bins and the time frames. The corresponding $i$-th speaker's noise $\hat{\mathbf{N}}_{i}(t, f)$ with the corresponding noise $\mathrm{cRF}_{\mathbf{N}_{i}}(t,f)$ could be estimated in the same way. The frame-wise $i$-th speaker's speech covariance is calculated as,
$$
\boldsymbol{\Phi}_{\mathbf{S}_{i}}(t, f)=\frac{\hat{\mathbf{S}}_{i}(t, f) \hat{\mathbf{S}}_{i}^{\mathrm{H}}(t, f)}{\sum_{t=1}^{T} \mathrm{cRM}_{\mathbf{S}_{i}}^{\mathrm{H}}(t, f) \mathrm{cRM}_{\mathbf{S}_{i}}(t, f)}
\tag{4}
$$
where $\mathrm{cRM}_{\mathbf{S}_{i}}(t, f)$ stands for the center mask of the $\mathrm{cRF}_{\mathbf{S}_{i}}(t, f)$. Then RNNs are used to directly learn the frame-level beamforming weights from the frame-wise covariance matrices, which can be formulated as
$$
\begin{aligned}
\mathbf{I}_{\mathrm{i}}(t, f) &=\left[\mathbf{\Phi}_{\mathbf{S}_{i}}(t, f), \mathbf{\Phi}_{\mathbf{N}_{i}}(t, f)\right] \\
\mathbf{w}_{1}(t, f), \ldots, \mathbf{w}_{\mathbf{C}}(t, f) &=\operatorname{RNN}\left(\left[\mathbf{I}_{1}(t, f), \ldots, \mathbf{I}_{C}(t, f)\right]\right)
\end{aligned}
$$
where $\mathbf{I}_{\mathrm{i}}(t, f)$ stands for the concatenation of $i$-th speaker's speech and noise covariance matrices, $\mathbf{w}_{\mathbf{i}}(t, f)$ denotes the $i$-th speaker's beamforming weights.

## 2.3. Proposed MIMO self-attentive RNN beamformer
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220816154528.png)
As shown in Fig. 1, the whole system consists of two parts, a complex-valued ratio filter(cRF) estimator and the proposed self-attentive RNN beamformer. Similar to GRNN-BF, the cRFs first help to estimate the speech and noise covariance matrices (Eq. (4)). Then the self-attentive RNN BF could predict the frame-level beamforming weights from the covariance matrices. Temporal self-attentive module, spatial self-attentive module and temporal-spatial self-attentive module are proposed to further improve the RNN-based BF to learn a better beamforming filter in this work.

### 2.3.1. Temporal self-attentive module
The input $\mathbf{Z}_1$ to the self-attentive module is processed by 3 linear transform layers (FFN) on the feature dim. The outputs of the FFN are represented as $\mathbf{Q},\mathbf{K},\mathbf{V}$.

The temporal self-attention calculates the attention matrix among frames.
Add & Norm: residual path (skip connection) and a layer normalization
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220816163345.png)
Finally, the output of the self-attention function is fed into another FFN to get the transformed output $\mathbf{Z}_2$.

### 2.3.2. Spatial self-attentive module
Spatial self-attentive module attends on the spatial channel dimension to learn the cross-channel info from the complex-valued speech and noise covariance matrices.

### 2.3.3. MIMO temporal-spatial self-attentive RNN BF
Combine the temporal and spatial self-attentive modules into the proposed temporal-spatial self-attentive RNN BF to model both cross-frame info and cross-channel correlations.

Similar to GRNN-BF baseline, the input to the self-attentive RNN BF is also the concatenated speech and noise covariance matrices of all speakers.

Si-SNR is used to train the model in an end-to-end mode. Each speaker's Si-SNR is averaged if that speaker exists in the utterance.

# Exp
## 3.1. System overview
cRF estimator is based on the Conv-TasNet with the fixed STFT encoder. The inputs to the cRF estimator include the 15-channel mic mixture and all speakers' DOA information (LPS, IPD, DF). The location guided DF calculates the cosine similarity between the $i$-th speaker's steering vector $v(\theta_i)$ and IPDs. With DF features, our model can extract the target speaker's speech from the specific DOA and it can avoid the speaker permutation problem.

# Results and Discussions
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220817105110.png)


