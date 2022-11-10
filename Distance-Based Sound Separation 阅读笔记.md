#! https://zhuanlan.zhihu.com/p/582104189
# Distance-Based Sound Separation 阅读笔记

# Abstract
Propose the novel task of *distance-based sound separation*, where sounds are separated based only on their distance from a single mic. We train a NN to separate near sounds from far sounds in single channel synthetic reverberant mixtures, relative to a threshold distance defining the boundary between near and far.

# 1. Intro
In our approach, we assume the user would like to hear any sounds that occur within a local region around them, and block sounds coming from farther away. A system that accomplishes this would allow the user to engage in normal conversations without the interference of a crowded environment, and without becoming deaf to non-speech sounds in their immediate area.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221104115313.png)

Intensity $I$ of the *direct path* component of the sound varies with distance $d$ according to $I \propto 1/d^2$. In contrast, in an enclosed area, the intensity of the sound's reverberation, is roughly independent of its distance to the mic. Thus the *direct-to-reverberation* ratio (DRR) decreases with distance, and has been shown to be a cue for human distance perception.

Absorption by the air is a freq-dependent effect of distance, and plays a strong role over distances beyond tens of meters. Other effects of distance come from spatial effects that could be detected using an array of microphones.

To validate the concept of distance-based sound separation, we train NN models using mixtures of near and far sounds, where the acoustic properties of the sounds have been emulated using an acoustic room simulator. This allows us to have ground-truth targets for the near and far signals relative to a distance threshold. Our results show that it is possible to perform separation solely based on distance.

# 2. Related Work
Beamforming uses multiple mics and generally uses the DOA of sources as a cue to separate them. Near-field BF methods can only distinguish distances of nearby sounds within a range using the spherical nature of acoustic wavefronts. In contrast, our method can learn to operate over a wider range of distances.

Distance estimation refs: Distance estimation and localization of sound sources in reverberant conditions using deep neural networks, Sound source localization in a multipath environment using convolutional neural networks, Sound source distance estimation using deep learning: an image classification approach, Joint direction and proximity classification of overlapping sound events from binaural audio

# 3. Methods
## 3.1. Model
STFT: 32 ms square-root Hann window with 16 ms hop

0.3-power-compressed magnitude of STFT is fed as input to $L$ layers of uni-directional LSTMs with $N$ units each. FC with sigmoid creates 2 masks $M_{near}$ and $M_{far}$ for the input STFT $Y$. To ensure STFT consistency [Differentiable consistency constraints for improved deep speech enhancement], the masked STFTs for near and far are each passed through inverse and forward STFT operations: $\hat{X}_{\text {near } \mid \text { far }}=\operatorname{STFT}\left\{\mathrm{iSTFT}\left(M_{\text {near } \mid \text { far }} \odot Y\right)\right\}$. See https://github.com/etzinis/fedenhance/blob/5f805dfaf6fb43e9e255803be25db282180eb8e7/fedenhance/experiments/utils/mixture_consistency.py.

The training loss is MSE between 0.3-power-compressed magnitude of target $X$ and estimate $\hat{X}$:
$$
\mathcal{L}(X, \hat{X})=\sum_{f, t}\left(\left|X_{f, t}\right|^{0.3}-\left|\hat{X}_{f, t}\right|^{0.3}\right)^2
$$
Total loss:
$$
0.8\mathcal{L}(X_{near}, \hat{X}_{near})+0.2\mathcal{L}(X_{far}, \hat{X}_{far})
$$
which encourages the model to focus on the performance of near targets, which is more likely to be desired as the output in a practical application.

## 3.2. Acoustic Simulation
Clean speech recordings are randomly assigned to the source locations within each room and convolved with the corresponding RIRs.

# 4. Exps
## 4.1. Data Preparation
Rooms are generated with dimensions varying from $3.0\times4.0\times2.13$ meters to $7.0\times8.0\times3.05$ meters.

To vary the number of sources, we apply a *source presence probability* (SPP) to each source, so that the total number of sources in a room varies from 0 to 5 with a distribution dependent on the chosen SPP.

## 4.2. Metrics
Use SI-SDRi. SI-SDR measures signal fidelity with respect to a reference signal while allowing for a gain mismatch:
$$
\operatorname{SI-SDR}(x, \hat{x})=10 \log _{10}\left(\|\alpha x\|^2 /\left\|\alpha x-\hat{x}^s\right\|^2\right),
$$
where $\alpha=\operatorname{argmin}_a\|a x-\hat{x}\|^2=x^T \hat{x} /\|x\|^2$.

SI-SDRi diverges when one of the targets is silent, because the non-silent target will be exactly equal to the input mixture and achieve $\infty$ dB SI-SDR, and the silent target will have $-\infty$ dB SI-SDR. This makes any improvement calculation meaningless. With a silent near target, we use a noise reduction metric, to measure how much of the far sound leaks into the silent near output:
$$
\text { NoiseReduction }\left(y, \hat{x}_{\text {near }}\right)=10 \log _{10}\left(\|y\|^2 /\left\|\hat{x}_{\text {near }}\right\|^2\right)
$$
which measures the power reduction of the separated output $\hat{x}_{near}$ relative to the input audio mixture power $y$.

# 5. Results
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221104225832.png)

## 5.1. Effect of distance threshold
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221104230634.png)
Noise reduction degrades with increased distance threshold.


## 5.2. Effect of model size
The SI-SDR differences between the smallest model and the largest model for distance thresholds of 0.8 m, 1.5 m and 3.0 m are 0.9 dB near / 1.2 dB far, 1.5 dB near / 1.6 dB far, and 1.8 dB near / 1.4 dB far, respectively. Note that for increasing model size, SI-SDRi improves more for higher distance thresholds.