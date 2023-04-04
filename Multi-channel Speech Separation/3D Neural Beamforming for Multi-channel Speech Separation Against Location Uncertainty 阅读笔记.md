# 3D Neural Beamforming Against Location Uncertainty 阅读笔记
# 3D Neural Beamforming for Multi-channel Speech Separation Against Location Uncertainty 阅读笔记

# Abstract
Multi-channel speech separation using speaker's directional information has demonstrated significant gains over blind speech separation. However, it has two limitations. First, substantial performance degradation is observed when the coming directions of two sounds are close. Second, the result highly relies on the precise estimation of the speaker's direction. To overcome these issues, this paper proposes 3D features and an associated 3D neural beamformer for multi-channel speech separation. Previous works in this area are extended in two important directions. First, the traditional 1D directional beam patterns are generalized to 3D. This enables the model to extract speech from any target region in the 3D space. Thus, spks with similar directions but different elevations or distances become separable. Second, to handle the spk location uncertainty, previously proposed *spatial* feature is extended to a new 3D *region* feature. The proposed 3D region feature and 3D neural beamformer are evaluated under an in-car scenario. Exp results demonstrated that the combination of 3D feature and 3D beamformer can achieve comparable performance to the separation model with ground truth spk location as input.

# Intro
Despite the impressive improvements achieved by direction-aware MC-TSS methods, the strong dependency on precise DOA estimation is not trivial. The DOA estimation error severely deteriorates separation performance especially when the azimuths of simultaneous speech are close.

First, the target spk is assumed to locate within a limited 3D region centered at the estimated location. The 3D setup is adopted to enable the MC-TSS model to distinguish sources with close azimuths via their different elevations and source-to-array distances.

Then, a 3D region feature is designed condition on the vertices and center of the region. Via a learning based attention module, the 3D region feature learns to aggregate and attend to diff spatial views of the region. The 3D region feature is served as the input to an AN-BF to advance beamforming weights estimation.

# 3D Feature
## 2.1. 3D spatial feature
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202303/20230315154343.png)
[3D Spatial Features for Multi-Channel Target Speech Separation, ASRU 2021]

location $\mathbf{l}=\{\theta,\phi,d\}$

For $p$-th mic pair
$$
S F_{t, f}(\mathbf{l})=\sum_p\left\langle\operatorname{IPD}_{t, f}^{(p)}, \operatorname{TPD}_f^{(p)}(\mathbf{l})\right\rangle
$$
where $\operatorname{IPD}_{t, f}^{(p)}=\angle \mathbf{Y}_{t, f}^{\left(p_1\right)}-\angle \mathbf{Y}_{t, f}^{\left(p_2\right)}, \operatorname{TPD}_f^{(p)}(\mathbf{l})=2 \pi f \tau^{(p)}(\mathbf{l}), \tau^{(p)}(\mathbf{l})=(d_{p_1}-d_{p_2})f_s/c$

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202303/20230315153254.png)

# 2.2. 3D region feature
3D spatial feature is sensitive to the location estimation error, which brings about extra burden for precise sound localization.

To account for the uncertainty of the location info, i.e., inaccurate source localization, array and camera miscalibration, this work makes an attempt to learn a robust model by posing the potential location deviations at the training stage.

A straightforward method is to intro random perturbations to the given azimuths, elevations and distances as the new input to the model, which may mislead the model to learn a broader main beam to tolerate the errors, therefore degrading the separation performance.

Assume each source is located within a limited 3D region (3D box in this work), the center of which is the estimated location of the target source.

$$
R F(\mathbf{l})=\sum_{i=1}^{\mathcal{L}} p\left(\mathbf{l}_i \mid S F\left(\mathbf{l}_1\right), \ldots, S F\left(\mathbf{1}_{\mathcal{L}}\right)\right) S F\left(\mathbf{1}_i\right)
$$
$\mathbf{l}_i$: $i$-th candidate location

$\mathcal{L}$: the total num of vertices including the center

$p\left(\mathbf{l}_i \mid S F\left(\mathbf{l}_1\right), \ldots, S F\left(\mathbf{1}_{\mathcal{L}}\right)\right)$: the posterior of the source existence at $\mathbf{l}_i$, estimated via an attention module optimized with the separation network.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202303/20230315162740.png)

# 3D All-Neural Beamforming
3 modules: 3D feature computation, mask estimation and all-neural beamforming

# Exp setup
## 4.1. Data preparation
In-car scenario: the main driver is always speaking, up to 3 spks.

Dual mic with 11.8 cm

RT60: [0.05, 0.7] s, SIR: [-6, 6] dB

height: [0.95, 1.15] m

3D box boundary is decided according to the head size (0.2 m) and the seat width of the car.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202303/20230315165151.png)

Conv-TasNet is served as the mask estimator.

7.36 v.s. 8.78 dB, 9.17 v.s. 9.77 dB (3D spatial feature alleviate spatial ambiguity issue)

3D-AN-BF ($\{\mathbf{l}_i\}^9_{i=1}-\{\mathbf{l}_i\}^9_{i=1}$) achieves comparable performance with GT-GT setup (9.79 v.s. 9.77 dB). In this way, the need for precise localization is mitigated.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202303/20230316105947.png)

# Conclusion
Proposed a 3D neural beamforming method for multi-channel separation to release the burden of precise source localization while accounting for the location uncertainty. A 3D region feature was designed to extract and selectively attend to different spatial views within a candidate region, and then integrated into an all-neural beamforming network.