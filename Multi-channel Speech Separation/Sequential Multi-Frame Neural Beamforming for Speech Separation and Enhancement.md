# Sequential Multi-Frame Neural Beamforming for Speech Separation and Enhancement 
## SLT 2021
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202302/20230227105518.png)
# Abstract
Introduces sequential neural beamforming, which alternates between neural network based spectral separation and beamforming based spatial separation.

Introduce a multi-frame beamforming method which improves the results significantly by adding contextual frames to the beamforming formulations.

# Intro
Leveraging multiple mics has great potential to improve separation, since the spatial relationship among mics provides complementary info to spectral cues exploited by monaural approaches.

Recently, a new paradigm has emerged as a promising alternative to conventional beamforming approaches: neural beamforming, where the key advance is to utilize the non-linear modeling power of DNN to identify T-F units dominated by each source for spatial covariance matrix computation. Unlike traditional approaches, neural beamforming methods have the potential to learn and adapt from massive training data, which improves their robustness to unknown positions and orientations of mics and sources, types of acoustic sources, and room geometry.

This paper explores alternating between spectral estimation using DNN-based masking and spatial separation using linear beamforming with a multichannel Wiener filter (MCWF), performing separate, beamform, separate, beamform, and separate. It is inspired by the single-channel sequential network of [Universal sound separation] and by the findings that better beamforming results can be used as extra features to improve spectral masking and vice versa. We also explore the effectiveness by incorporating multi-frame context during beamforming.

# Methods
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202302/20230227155654.png)

Each spectral masking stage uses an improved TDCN++. The first stage performs single-channel processing to estimate each source via T-F masking. The estimated sources are then used to compute statistics for time-invariant or time-varying beamforming. The next masking stage combines spectral and spatial info by taking in the mixture and beamformed results for post-filtering. This sequence is then repeated several times.

$$
\hat{X}_{\mathrm{MN} i}^{(s)}=\hat{A}_i^{(s)} \odot Y_{\mathrm{ref}}
$$

$\hat{A}_i^{(s)}$ is the mask estimate produced by the $i$-th TDCN++. $I=3$ stages in the sequence. For $i>1$, the network input is the concatenation of the mixture mag with those of all beamformed source estimates, $\hat{X}_{\mathrm{BF} i-1}^{(s)}$.

## 3.2. Multi-frame multichannel Wiener filter

By stacking multiple frames, the beamformer can have more contextual info and degrees of freedom for better noise suppression.

The multi-frame mixture covariance matrix is estimated as
$$
\hat{\boldsymbol{\Phi}}_f^{(y)}=\frac{1}{T} \sum_{t=1}^T \overline{\mathbf{Y}}_{t, f} \overline{\mathbf{Y}}_{t, f}^H,
$$
and $\hat{\boldsymbol{\Phi}}_{i, f}^{(s)}$ is the source covariance matrix computed as
$$
\begin{aligned}
& \hat{\boldsymbol{\Phi}}_{i, f}^{(s)}=\frac{1}{T} \sum_{t=1}^T \hat{A}_{i, t, f}^{(s)} \overline{\mathbf{Y}}_{t, f} \overline{\mathbf{Y}}_{t, f}^H, \\
& \hat{A}_{i, t, f}^{(s)}=\frac{\left|\hat{X}_{\mathrm{MN} i, t, f}^{(s)}\right|^2}{\sum_{s^{\prime}=1}^S\left|\hat{X}_{\mathrm{MN}, t, t, f}^{\left(s^{\prime}\right)}\right|^2} . \\
&
\end{aligned}
$$

$$
\hat{\mathbf{w}}_{i, f}^{(s)}=\left(\hat{\boldsymbol{\Phi}}_f^{(y)}\right)^{-1} \hat{\boldsymbol{\Phi}}_{i, f}^{(s)} \mathbf{u}_{\mathrm{ref}}
$$

each stage $i$, for each source $s$;
$\mathbf{u}_\mathrm{ref}$ is a one-hot vector with the coeff corresponding to the ref mic at the center frame set to one.