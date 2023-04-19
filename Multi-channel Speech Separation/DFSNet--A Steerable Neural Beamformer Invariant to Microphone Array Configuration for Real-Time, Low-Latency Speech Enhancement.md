#! https://zhuanlan.zhihu.com/p/622772193
# DFSNet: A Steerable Neural Beamformer Invariant to Mic Array Config for Real-Time SE 阅读笔记
# DFSNet: A Steerable Neural Beamformer Invariant to Microphone Array Configuration for Real-Time, Low-Latency Speech Enhancement 阅读笔记

# Abstract
Filter-and-sum methods define the target signal with respect to a reference channel. This not only complicates formulation in reverberant conditions but the network, which must have a mechanism to infer what the ref channel is. To address these issues, Delay Filter-and-Sum Network (DFSNet), a steerable neural beamformer invariant to mic number and array geometry for causal speech enhancement is proposed.

In DFSNet, acquired signals are first steered toward the speech source direction prior to the FS operation, which simplifies the task into the estimation of delay-and-summed reverberant clean speech. The proposed model is designed to incur low latency, distortion, memory and computational burden, giving rise to high potential in hearing aid applications.

# 1. Intro
Time domain methods [Temporal-spatial neural filter, Ux-net] are an increasingly popular class among neural beamformers because of their high potential in latency-demanding applications, such as hearing aids. However, unless retrained, the proposed networks rarely provide invariance to mic array configuration (mic num and array geometry), an attribute of special importance in ad-hoc array scenarios.

Consistent with other multi-channel methods, FaSNet specifies its target signal w.r.t the ref mic. Thus, when trained for SE, the target signal of FaSNet is the clean reverberant speech at the ref mic. However, this formulation introduces two complications. (1) The model needs to somehow learn how to combine the different-channel signals to both reduce noise as well as reconstruct the direct path and reverberant components of speech at a reference microphone. (2) As a consequence of array geometry invariance, special processing with respect to the reference microphone must be introduced, otherwise the model has no means to infer what the reference microphone is.

Motivated by the above observations, this study proposes DFSNet (Delay-Filter-and-Sum), which operates in a framewise manner and follows a linear signal model analogous to freq-domain FS beamforming. In DFSNet, time-domain waveforms are first delayed by a set of integer and fractional delay finite impulse response (FIR) filters toward the speech source direction. Delayed signals are then converted into a latent space representation through a linear transformation. Next, masks for each channel are estimated by a stack of recurrent channel interaction (RCI) blocks, which efficiently combine recurrent processing with a channel interaction (CI) technique similar to TAC. Finally, FS is applied in the latent space representation followed by a linear transformation to convert the result back to the time domain. As a consequence of signal delay prior to FS, the target signal of DFSNet is defined as the delay-and-summed reverberant clean speech. With this approach, DFSNet simplifies the task into learning how to collectively reduce noise at individual channels; avoids specifying a reference microphone; and allows steering to different directions without retraining.

# 2. Problem Formulation
TDOA between $\mathbf{m}_1$ (furthest mic position) and $\mathbf{m}_i$
$$
\tau_i=f s^{-1}\left(\left\|\mathbf{u}-\mathbf{m}_1\right\|-\left\|\mathbf{u}-\mathbf{m}_i\right\|\right) \tag{2}
$$
$\mathbf{u}, \mathbf{m}_c$: 3D positions of the speech source and $c$-th mic.

$\widetilde{\tau}_i$: a known positive estimate of $\tau_i$

We can align the acquired signals toward an approximate direction of the speech source by
$$
\mathbf{y}_i^a=\left(\mathbf{h}_{F_i} * \mathbf{h}_{D_i}\right) * \mathbf{y}_i, \quad i=2,3, \ldots, C \tag{3}
$$
where $\mathbf{h}_{D_i}$ and $\mathbf{h}_{F_i}$ are causal integer and fractional delay FIR filters, and $*$ denotes convolution. 

$D_i=\lfloor\widetilde{\tau}_i\rfloor$, $F_i=\widetilde{\tau}_i-D_i$ specify the sample delay of a filter $\mathbf{h}$.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230417214242.png)
$\mathbf{y}=\mathbf{x}+\mathbf{v}$: mixture = speech + noise

# 3. DFSNet
Three stages: encoder, filter estimator, decoder
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230418102158.png)
## 3.1. Encoder
At the encoder, input channels are first aligned applying (3) and (4) to produce utterances $\mathbf{y}_c^a$ of length $T$ samples. Each utt is segmented into $K$ sequential overlapping frames of length $L$ samples and 50% overlap.

## 3.2. Filter Estimator
The filter estimator estimates channel and time-varying filters given by mask vectors $\hat{\mathbf{m}}_{c,k}$ for application in the latent space. In this module, each $N$-dimensional latent space representation $\mathbf{z}_{c,k}$ is first normalized applying sliding window layer normalization to reduce variability and speed up training, followed by stacked RCI blocks and a sigmoid to ensure non-negative masks.

## 3.3. Decoder
estimated masks and latent space representations are multiplied and summed across the channel dimension followed by transformation back to the time domain by an FC layer with weights $\mathbf{B}_d$ and no bias:
$$
\hat{\mathbf{x}}_{D S, k}=\left(\frac{1}{C} \sum_{c=1}^C \hat{\mathbf{m}}_{c, k} \odot \mathbf{z}_{c, k}\right) \mathbf{B}_d
$$
where $\odot$ denotes element-wise product. $\hat{\mathbf{x}}_{D S}$ is then reconstructed by the overlap-add operation.

The proposed encoder/decoder operations follow a linear signal model analogous to freq-domain FS beamforming, with the difference that instead of STFT, we apply forward and inverse transformations learned by the network. 

## 3.4. Sliding Window Layer Normalization (sLN)
sLN is similar to cumulative LN (cLN) with the difference that normalization is performed over a sliding window of fixed size $R$ rather than cumulatively, thus allowing for better adaptation in applications where signal statistics can drastically change over time.

The sLN is applied at each channel independently.

## 3.5. Recurrent Channel Interaction (RCI)
RCI block combines GRUs and a CI technique similar to that in TAC blocks. The aim is to gain spatio-temporal context awareness, necessary for estimation of beamforming filters, w/o sacrificing invariance to mic num and array geometry. 

## 3.6. Local and Global Processing
local: intra-channel operations are shared across channels
global: inter-channel operations

# 4. Exps
utts from WHAM! to simulate noisy speech captured by a mic array of arbitrary num of mics and geometry in a reverberant room.

SI-SDR

# 5. Results and Analysis
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230418102751.png)

The target signal is the reverberant clean speech at a reference microphone, selected as the closest microphone to source due to its highest SNR.

We notice that DFSNet underperforms in SI-SDR when evaluated with respect to a reference microphone, especially as the microphone number increases. However, when it comes to perception and intelligibility, DFSNet outperforms all causal methods by a significant margin. Additionally, when evaluated with respect to its target signal, DFSNet outperforms MVDR in all cases, and, with just two microphones, attains comparable performance to the noncausal DPRNN-TasNet. Moreover, when the number of microphones increases, DFSNet approaches and in certain cases exceeds the performance of the noncausal FaSNet-TAC. We further note that DFSNet incurs only a fraction of memory and computational cost of SOTA models, which is largely attributed to the proposed feature partitioning scheme in RCI blocks.

## 5.2. Ablation study

# 6. Conclusion
Proposed DFSNet, a steerable neural beamformer invariant to microphone array configuration for real-time SE. In contrast to conventional FS methods, DFSNet performs a channel alignment procedure prior to applying the FS operation, which simplifies the beamforming task into the estimation of DS clean reverberant speech. The proposed model incurs low latency, distortion, and memory and computational burden, making it suitable for hearing aid applications. Comparison with SOTA revealed that DFSNet outperforms causal methods in perception and intelligibility by a large margin. Additionally, we noted that DFSNet outperforms MVDR and approaches the performance of the noncausal FaSNet-TAC.