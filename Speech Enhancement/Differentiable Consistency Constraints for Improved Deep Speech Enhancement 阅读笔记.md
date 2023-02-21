#! https://zhuanlan.zhihu.com/p/581049927
# Differentiable Consistency Constraints for Improved Deep Speech Enhancement 阅读笔记
## [ICASSP 2019]
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221117113929.png)
# Abstract
Current masking approaches often neglect two important constraints: STFT consistency and mixture consistency. Without STFT consistency, the system's output is not necessarily the STFT of a time-domain signal, and without mixture consistency, the sum of the estimated sources does not necessarily equal to the input mixture.

We show that STFT consistency and mixture consistency can be jointly imposed by adding simple differentiable projection layers to the enhancement network. These layers are compatible with real or complex-valued masks.

# 1. Intro
Existing masking-based DNN approaches have two deficiencies. First, the loss func used to train the enhancement network is typically measured on the masked noisy STFT. The problem with this is that applying an arbitrary mask does not produce a consistent STFT, in the sense that the masked STFT could not be computed from any real-valued time-domain signal. Second, some approaches that use a real-valued mask and all approaches that use a complex-valued mask do not leverage the basic assumption given by the model: that the separated sources should add up to the original mixture signal.

We show that STFT and mixture consistency can be enforced by adding simple end-to-end layers in the enhancement network. These two techniques can be combined with any masking-based speech enhancement model to improve performance.

# 3. Consistency
## 3.1. STFT consistency
When a STFT uses overlapping frames, applying a mask to the STFT of a mixture signal, whether the mask is real or complex-valued, does not necessarily produce a consistent STFT. A consistent STFT $\mathbf{X}$ is one such that there exists a real-valued time-domain signal $\mathbf{x}$ satisfying $\mathbf{X}=\mathcal{S}\{\mathbf{X}\}$. A STFT $\mathbf{X}$ is consistent if it satisfies
$$
\mathbf{X}=\mathcal{S}\left\{\mathcal{S}^{-1}\{\mathbf{X}\}\right\}=\mathcal{P}_{S}\{\mathbf{X}\}\tag{3}
$$
where $\mathcal{P}_{S}$ refers to the projection performed by the sequence of inverse and forward STFT operations.

If the masked STFT is simply used in a loss func like MSE between magnitudes of the masked and ground-truth spectrograms, the magnitudes $\left|\hat{\mathbf{X}}_j\right|=\left|\mathbf{M}_j \odot \mathbf{Y}\right|$ do not necessarily correspond to the STFT magnitudes of the reconstructed time-domain signal, $\left|\mathcal{S}\left\{\hat{\mathbf{x}_{j}}\right\}\right|$. Thus, the loss func will not be accurately measuring the spectral magnitude of the estimate.

### 3.1.1. Illustration of STFT consistency
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221104103927.png)

Compute an oracle phase-sensitive mask as
$$
\frac{\left|S_{f, t}\right|}{\left|S_{f, t}+V_{f, t}\right|} \cdot \cos \left(\angle S_{f, t}-\angle Y_{f, t}\right)\tag{4}
$$
This mask is applied to the noisy STFT to create a masked STFT. Then, we use the projection (3) to create a consistent STFT of the enhanced speech.

The masked STFT differs substantially from the consistent STFT, which indicates the importance of STFT consistency: if a NN training loss is measured on the masked magnitude instead of the consistent magnitude, then the loss is not looking at the actual spectrogram of the enhanced time-domain signal.

### 3.1.2. Backpropagating through STFTs
A natural way to enforce STFT consistency is to simply project estimated signals using the constraint (3). Since the forward and inverse STFT operations are linear transforms implemented in TensorFlow, this projection can simply be treated as an extra layer in the network.

## 3.2. Mixture consistency
Mixture consistency constraint has been enforced using real-valued masks that are made to sum to one across the sources. However, when the masks are complex-valued, constraining these masks to sum to one across sources is too restrictive, since complex masks might to need to modify the phase differently for different sources in the same T-F bin.

### 3.2.1. Backpropagating through mixture-consistent projection
To ensure mixture consistency without putting explicit constraints on the masks, we project the masked estimates to the nearest points on the subspace of mixture-consistent estimates. To do this, we solve the following optimization problem for each TF bin:
$$
\begin{array}{ll}
\underset{\underline{\mathbf{x}}_{f, t} \in \mathbb{C}^J}{\operatorname{minimize}} & \frac{1}{2} \sum_j\left|\underline{X}_{j, f, t}-\hat{X}_{j, f, t}\right|^2 \\
\text { subject to } & \sum_j \underline{X}_{j, f, t}=Y_{f, t}
\end{array}\tag{6}
$$
$\underline{X}_{j, f, t}$: the mixture-consistent TF bin for source $j$.

$\hat{X}_{j, f, t}$: masked / estimated STFT

Using the method of Lagrange multipliers and defining the estimated mixture $\hat{Y}_{f, t}=\sum_j\hat{X}_{j, f, t}$ yields the following update, which is a simple projection that can be added as a layer in the network and backpropagated through:
$$
\underline{X}_{j, f, t}=\hat{X}_{j, f, t}+\frac{1}{J}\left(Y_{f, t}-\hat{Y}_{f, t}\right) . \tag{7}
$$
Above is one solution to the optimization problem Eq. (6).

Replace $\hat{X}_{j, f, t}$ with $\underline{X}_{j, f, t}$ to compute loss function.

See https://github.com/etzinis/fedenhance/blob/5f805dfaf6fb43e9e255803be25db282180eb8e7/fedenhance/experiments/utils/mixture_consistency.py.

### 3.2.2. Weighted mixture-consistent projection with uncertainty
Assume that in each freq bin, each estimated source can be modeled as a zero-mean complex-valued circular Gaussian with variance $v_{j,f,t}$.

A weighted version of the problem (6) is
$$
\begin{array}{ll}
\underset{\underline{\mathbf{x}}_{f, t} \in \mathbb{C}^J}{\operatorname{minimize}} & \frac{1}{2} \sum_j\frac{1}{v_{j,f,t}}\left|\underline{X}_{j, f, t}-\hat{X}_{j, f, t}\right|^2 \\
\text { subject to } & \sum_j \underline{X}_{j, f, t}=Y_{f, t}
\end{array}\tag{8}
$$
Using the method of Lagrange multipliers yields
$$
\underline{X}_{j, f, t}=\hat{X}_{j, f, t}+\frac{v_{j,f,t}}{\sum_{j^{\prime}}v_{j^{\prime},f,t}}\left(Y_{f, t}-\hat{Y}_{f, t}\right) . \tag{9}
$$
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221106113613.png)

# 4. Exps

## 4.2. Model architecture and training
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221105232838.png)

Input features are power-compressed STFTs, computed as $X^{0.3}:=|X|^{0.3}e^{j\angle X}$.

$$
\begin{aligned}
L=\sum_{j=1}^2 z_j \sum_{f, t} & {\left[\left(\left|X_{j, f, t}\right|^{0.3}-\left|\hat{X}_{j, f, t}\right|^{0.3}\right)^2\right.} \\
&\left.+0.2 \cdot\left|X_{j, f, t}^{0.3}-\hat{X}_{j, f, t}^{0.3}\right|^2\right]
\end{aligned}\tag{11}
$$
where $z_1=0.8$ is the speech loss weight and $z_2=0.2$ is the noise loss weight.

Why 0.3-power-compressed?
: Power-law compression with power 0.3 approximates a log function while avoiding $-\infty$ at 0, partially equalizing the importance of quieter sounds relative to loud ones [Exploring Tradeoffs in Models for Low-latency Speech Enhancement] [R. F. Lyon, Human and machine hearing. Cambridge University Press, 2017, Chapter 2].

Our proposed consistency constraints are compatible with both real and complex-valued masks. For real-valued masking, the network predicts a single scalar value through a sigmoid nonlinearity for each TF bin, and the noisy phase is used for reconstruction. To perform complex-valued masking, the network predicts the real and imag parts of a complex-valued mask through a hyperbolic tangent (tanh) nonlinearity. This mask is multiplied with the complex noisy STFT, then reconstructed.

We consider 2 types of weighting schemes to implement weighted mixture consistency:
1. Weights $v_{j,f,t}$ are the squared estimated source magnitudes, $|\hat{X}_{j,f,t}|^2$. This has the advantage of not adding any correction signal to TF bins that have a low mag, which helps when there is only one signal active within a TF bin.
2. Weights are learned by the network. Sigmoid produces weights $w_{1,f,t}$ for the speech source, and $w_{2,f,t}=1-w_{1,f,t}$ for the noise source.

## 4.3. Results
Models that use both STFT and mixture consistency constraints almost always outperform models that do not use these constraints.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221106182026.png)

Phase prediction provides a slight improvement in performance, especially when using a complex mask for phase prediction at lower input SNRs.

Improves 0.7 dB over baseline when using complex masking, STFT consistency and weighted mixture consistency with learned weights.

# 5. Conclusion
Shown that the simple addition of differentiable neural network layers can be used to enforce STFT and mixture consistency on source estimates of an audio source separation network.