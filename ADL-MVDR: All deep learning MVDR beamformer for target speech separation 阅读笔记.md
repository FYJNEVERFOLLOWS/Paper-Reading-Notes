#! https://zhuanlan.zhihu.com/p/554644596
# ADL-MVDR: All deep learning MVDR beamformer for target speech separation 阅读笔记

# Abstract
Purely NN-based speech separation systems cause nonlinear distortion that is harmful to ASR. The conventional mask-based MVDR beamformer can be used to minimize the distortion, but comes with high level of residual noise. Furthermore, the matrix operations (e.g., matrix inversion) involved in the conventional MVDR solution are sometimes numerically unstable when jointly trained with NNs.

Propose ADL-MVDR (all deep learning) framework, where the matrix inversion and eigenvalue decomposition are replaced by two RNNs, to resolve both issues at the same time. The proposed method can greatly reduce the residual noise while keeping the target speech undistorted by leveraging on the RNN-predicted frame-wise beamforming weights. 

# Intro
MVDR filters aim to reduce the noise while keeping the target speech undistorted. Existing MVDR systems with NN-based TF mask estimator still suffer from residual noise problems since chunk- or utt- level beamforming weights are not optimal for noise reduction.

3 main contributions:
1. Propose all deep learning MVDR framework (ADL-MVDR) where the ADL-MVDR can be jointly trained stably with the front-end filter estimator for predicting frame-level beamforming weights.
2. Use two RNNs to replace the matrix inversion and PCA involved in the MVDR solution, instead of utilizing the traditional mathematical approach.
3. Instead of using the per T-F bin mask, adopt a complex ratio filtering method (cRF) to further stabilize joint training process and estimate the covariance matrices of target speech and noise more accurately.

# Conventional mask-based MVDR BF
Noisy speech mixture $y=[y_1,y_2,...,y_M]^T$

Clean speech $\mathbf{s}$

Interfering noise $\mathbf{n}$

The separated speech $\hat{\mathbf{s}}_{\mathbf{M V D R}}(t, f)$ can be obtained as
$$
\hat{\mathbf{s}}_{\mathbf{M V D R}}(t, f)=\mathbf{h}^{\mathrm{H}}(f) \mathbf{Y}(t, f)
$$
where $\mathbf{h}(f)$ represents the MVDR weights at freq index $f$ and $^\mathrm{H}$ stands for the Hermitian operator. The goal of the MVDR BF is to minimize the power of the noise while keeping the target speech undistorted, which can be formulated as 
$$
\mathbf{h}_{\mathbf{M V D R}}=\underset{\mathbf{h}}{\arg \min } \mathbf{h}^{\mathrm{H}} \boldsymbol{\Phi}_{\mathrm{NN}} \mathbf{h} \quad \text { s.t. } \quad \mathbf{h}^{\mathrm{H}} \boldsymbol{v}=\mathbf{1}
$$
$\boldsymbol{\Phi}_{\mathrm{NN}}$ stands for the covariance matrix of the noise power spectral density (PSD) and $v(f)$ denotes the steering vector of the target speech.

We focus on the MVDR solution that is based on the steering vector, which can be derived by applying PCA on the speech covariance matrix.
$$
\mathbf{h}(f)=\frac{\boldsymbol{\Phi}_{\mathrm{NN}}^{-1}(f) \boldsymbol{v}(f)}{\boldsymbol{v}^{\mathrm{H}}(f) \boldsymbol{\Phi}_{\mathrm{NN}}^{-1}(f) \boldsymbol{v}(f)}
$$
cRM can be used to estimate the target speech accurately with less amount of phase distortion, which benefits human listeners.

In this case, the estimated speech $\hat{\mathbf{S}}_{\mathrm{cRM}}$ and covariance matrix of the speech PSD $\boldsymbol{\Phi}_{\mathrm{SS}}$ can be computed as
$$
\begin{aligned}
\hat{\mathbf{S}}_{\mathrm{cRM}}(t, f) &=\mathbf{M}_{\mathrm{S}}(t, f) * \mathbf{Y}(t, f) \\
\boldsymbol{\Phi}_{\mathrm{SS}}(f) &=\frac{\sum_{t=1}^{T} \hat{\mathbf{S}}_{\mathrm{cRM}}(t, f) \hat{\mathbf{S}}_{\mathrm{cRM}}^{\mathrm{H}}(t, f)}{\sum_{t=1}^{T} \mathbf{M}_{\mathrm{S}}^{\mathrm{H}}(t, f) \mathbf{M}_{\mathrm{S}}(t, f)}
\end{aligned}
$$
Note that the covariance matrix $\Phi$ derived here is on the utt-level, leading to utt-level beamforming weights which are not optimal for noise reduction on each frame.

# ADL-MVDR BF
Implement 2 GRU-based networks to replace the matrix inversion and PCA for estimating frame-level beamforming weights.

## 3.1. cRF for covariance matrix estimation
For each T-F bin, the cRF is applied to its $(2K+1)\times(2L+1)$ nearby T-F bins as
$$
\begin{aligned}
\hat{\mathbf{S}}_{\mathrm{cRF}}(t, f) &=\sum_{\tau_{1}=-L}^{L} \sum_{\tau_{2}=-K}^{K} \mathbf{F}_{\mathrm{S}}\left(t+\tau_{1}, f+\tau_{2}\right) \\
& * \mathbf{Y}\left(t+\tau_{1}, f+\tau_{2}\right) \\
\boldsymbol{\Phi}_{\mathrm{SS}}(t, f) &=\frac{\hat{\mathbf{S}}_{\mathrm{cRF}}(t, f) \hat{\mathbf{S}}_{\mathrm{cRF}}^{\mathrm{H}}(t, f)}{\sum_{t=1}^{T} \mathbf{M}_{\mathrm{S}}^{\mathrm{H}}(t, f) \mathbf{M}_{\mathrm{S}}(t, f)}
\end{aligned}
\tag{6}
$$
$\hat{\mathbf{S}}_{\mathrm{cRF}}$: the estimated speech

$\mathbf{F}_{\mathrm{S}}$: complex ratio filter

The cRF is equivalent to $(2K+1)\times(2L+1)$ number of cRMs that each applied to the corresponding shifted version (along time and freq axes) of the noisy spectrogram.

$\mathbf{M}_{\mathrm{S}}(t,f)$: the center mask of the cRF

Note that we do not sum over the time dimension of $\boldsymbol{\Phi}_{\mathrm{SS}}$ in order to preserve the frame-level temporal information.

## 3.2. RNNs for replacing matrix inversion and PCA in MVDR
Propose to replace the mathematical derivation of the steering vector and the inversion of noise covariance matrix with two GRU-Nets. These GRU-Nets will learn the steering vector and the matrix inversion through back propagation, i.e.,
$$
\begin{aligned}
\hat{\boldsymbol{v}}(t, f) &=\text { GRU-Net }_{\boldsymbol{v}}\left(\boldsymbol{\Phi}_{\mathrm{SS}}(t, f)\right), \\
\hat{\boldsymbol{\Phi}}_{\mathrm{NN}}^{-1}(t, f) &=\mathbf{G R U}_{\mathbf{R}} \mathbf{N e t}_{\mathrm{NN}}\left(\boldsymbol{\Phi}_{\mathrm{NN}}(t, f)\right),
\end{aligned}
$$
where the real and imag parts of the complex-valued covariance matrix $\boldsymbol{\Phi}$ are concatenated together as input to the GRU-Net. Leveraging on the temporal structure of RNNs, the model recursively accumulates and updates the covariance matrix for each frame.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220812205409.png)
As shown in Fig. 1, the output of each GRU-Net is fed into a linear layer to obtain the final real and imag parts of the complex-valued covariance matrix or steering vector. Then, we compute the frame-level ADL-MVDR weights as
$$
\begin{aligned}
\hat{\boldsymbol{v}}(t, f) &=\text { GRU-Net }_{\boldsymbol{v}}\left(\boldsymbol{\Phi}_{\mathrm{SS}}(t, f)\right), \\
\hat{\boldsymbol{\Phi}}_{\mathrm{NN}}^{-1}(t, f) &=\mathbf{G R U}_{\mathbf{R}} \mathbf{N e t}_{\mathrm{NN}}\left(\boldsymbol{\Phi}_{\mathrm{NN}}(t, f)\right),
\end{aligned}
$$

Finally, the ADL-MVDR enhanced speech is obtained
$$
\hat{\mathbf{s}}_{\mathbf{ADL-M V D R}}(t, f)=\mathbf{h}^{\mathrm{H}}(f) \mathbf{Y}(t, f)
\tag{9}
$$

# Exp
## 4.1. System overview
Extract the log-power interaural phase difference (IPD) and spectra (LPS) features from the 15-ch mic recorded mixture.

The DOA is roughly estimated using the 180° camera, then the location guided directional feature (DF) is extracted.

The DF is merged with the IPD and LPS features before fed into the audio encoding blocks.

## 4.2. Exp setup
512-point FFT with 32ms Hann win and 16ms step.

The size of cRF is $3\times3$ (i.e., $2\mathrm{K}+1$ and $2\mathrm{L}+1$ are 3)

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220812210757.png)
multi-tap: $[t,t-1]$

For purely NN-based systems, although they perform well across objective metrics, they perform poorly in ASR system due to large amount of distortion.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220812211154.png)
As shown in Fig. 2, ADL-MVDR alleviate the residual noise while keeping the target speech undistorted.

# Conclusion
Propose all deep learning MVDR method to recursively learn the spatio-temporal filtering for multi-channel target speech separation.
