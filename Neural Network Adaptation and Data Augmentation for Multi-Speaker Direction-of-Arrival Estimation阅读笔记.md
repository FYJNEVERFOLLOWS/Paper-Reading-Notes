# Introduction

## A. Motivation and Challenges

Signal processing approaches are built around analytical solutions. These solutions rely on assumptions about the acoustic environments, like known transfer functions or steering vectors, free-field anechoic sound propagation, high SNR, spatial white noise, or a known number of sources. (These assumptions may not hold well in real-world scenarios.)

There are often multiple simultaneous sound sources in the environments. The discrepancy between assumptions and reality may lead to significant performance degradation. Sophisticated modeling of the complex environments may mitigate the problem, but it is not clear how to generalize it as exhaustive modeling of all environments is not possible.

In the learning-based approaches, the difficulties have been shifted from modeling the complex environments in the signal processing approaches to the need of collecting a sufficient number of training data covering all variabilities (including various sound classes, samples per class, source locations, reverberation, noises, and solid objects in the scenes) in the test environment.

Annotating audio recordings with the ground truth labels is particularly costly.

A popular way of obtaining training data for sound source localization is by acoustic simulation.

Domain adaptation, which uses both simulated and real data, may be applied to SSL.

## B. Goals and Contributions

The contributions of this paper are:

* Propose a multi-source DOA estimation framework with domain adaptatoin so that the data collection workload can be significantly reduced.
* Propose a weakly-supervised adaptation scheme that minimizes the distance in the output coding space between the network output and all the predictions consistent with the weak labels.
* The weakly-supervised adaptation scheme is extended through data augmentation, which improves the performance of the weakly-supervised adaptation.

# Related Work

## A. NN-based Sound Source Localization

Different approaches differ in their input representation, output coding as well as their network structures.

More recent studies have shown that low-level signal representation without explicit feature extraction, whether in the time or time-frequency domains, can allow the networks to learn to extract the most informative high-level features for SSL.

Few of studies on DL based SSL clearly address issues related to the high cost of data collection, especially by applying domain adaptation to models trained with simulated data.

## B. Domain Adaptation

Domain adaptation explores how the knowledge from a dataset (source domain) can be exploited to help build machine learning models on another set (target domain). Domain adaptation approaches include re-weighting samples so that the loss function on the source samples are corrected to approximate that on the target domain.

# DOA Estimation Model

![image-20211026171541429](https://tva1.sinaimg.cn/large/008i3skNly1gvstyyv18qj30mu0bzq4i.jpg)

## A. Overview

The model consists of three parts: feature representation, output coding, and network architecture.

## B. Network Input

The network input comprises the real and imaginary parts of the time-frequency domain signal.

In contrast to high- level features extraction, such a representation retains all the information of the signal and allows the network to implicitly extract informative features for localization, which potentially include both inter-channel (通道间) cues (i.e. level/phase difference) and intra-channel (通道内) cues (i.e. spectral features).

用 STFT 的另一个好处是说，语音信号在时频表示上稀疏，这样的特征，有助于网络学习定位多个声源的情况。

We prepare the network input as follows:

First divide the 4-channel audio into 170 ms long segments (8192 samples in 48 kHz recordings). [This segment size provides a good balance between the amount of information and the time resolution. Such a short input segment is suitable for real-time applications]

Compute the STFT of the segments with a frame size of 43 ms (2048 samples) and 50% overlap. Thus, there are seven frames in each segment. 

Only use the frequency bins between 100 and 8000 Hz, so that the number of frequency bins is reduced to 337. 

Take the real and imaginary part of the complex values instead of the phase and power, so that we avoid the discontinuity problem of the phase at $\pi$ and $-\pi$.

Eventually, the dimension of the input vector is 7 × 337 × 8.

## C. Output Coding

The spatial spectrum coding:

A spatial spectrum is a function of the DOA and its value indicates how likely there is a sound source for a given DOA.

Thus, the localization problem becomes a spatial spectrum regression problem.
$$
o^{*}(y)= \begin{cases}\max _{\phi^{\prime} \in y}\left\{e^{-d\left(\phi_{i}, \phi^{\prime}\right)^{2} / \sigma^{2}}\right\} & \text { if }|y|>0 \\ 0 & \text { otherwise }\end{cases} \tag{2}
$$
![image-20211026190959528](https://tva1.sinaimg.cn/large/008i3skNly1gvsx9wn682j30mx08bmxz.jpg)

$y$ : label, a set of locations 

$|y|$: the number of sources

$\phi'$: one ground truth DOA

$\sigma$: beam width

H.L. Van Trees, Detection, Estimation, and Modulation Theory, Optimum ArrayProcessing, John Wiley & Sons, New-York, USA, 2004.

波束宽度是主瓣两侧的两个最低值之间的间距（即主瓣的零点之间的宽度）。
$$
\sigma = \theta_{B W}=2 \sin ^{-1}\left(\frac{c}{M d f}\right)
$$
M: num_mics, d: 麦克风间距, c: sound speed



Decode when inferencing

When the number of sources $z$ is unknown, the peaks above a given threshold $\xi$ are taken as predictions:
$$
\hat{y}(o ; \xi)=\left\{\phi_{l}: o_{l}>\xi \quad \text { and } \quad o_{l}=\max _{d\left(\phi_{i}, \phi_{l}\right)<\sigma_{n}} o_{i}\right\}
$$
When the number of sources $z$ is known, the $z$ highest peaks are taken as predictions:
$$
\hat{y}(o ; z)=\left\{\phi_{l}: \text { among the } z \text { greatest } o_{l}=\max _{d\left(\phi_{i}, \phi_{l}\right)<\sigma_{n}} o_{i}\right\} \text {. }
$$
$o = f_\theta(x)$ is the network output.

## D. Network Architecture

Fully-convolutional neural network structure.

CNN facilitates weight sharing for deep neural network models, thus reducing the overall number of parameters as well as the risk of overfitting.

Recurrent CNN can leverage the context information. (recurrent structure may introduce additional computational cost)

Our network comprises two parts, which convolve along different axes.

In the first part, the network convolves along the time and frequency axes. Specifically, it includes two layers of strided convolution in the frequency axis for downsampling as well as feature extraction, five residual blocks for the extraction of higher level features, and a layer projecting the features to the DOA space. The residual connection allows the construction of very deep neural network models, and therefore increases their capabilities at extracting high-level features. The output of the first part of the network is time-frequency local, meaning that each output value is derived from a local time-frequency region of the input.

In the second part, the network convolves along the DOA axis. It aggregates features in the neighboring directions across all the time-frequency bins (global), and outputs a spatial spectrum.

![image-20211026193708696](https://tva1.sinaimg.cn/large/008i3skNly1gvsy23r1t5j30fp0o6abz.jpg)

The first part (green) applies convolution along the time and frequency axes, and the second part (blue) applies convolution along the DOA axis.

## E. Two-stage Training

The goal of training is to make the network regress the ideal spatial spectrum with the MSE loss:
$$
\mathcal{L}\left(f_{\theta}(x), y\right)=\left\|f_{\theta}(x)-o^{*}(y)\right\|_{2}^{2}\tag{5}
$$
In the first stage, we train the first part of the network, by considering its output as the short-term narrow-band predictions of the spatial spectrum.

The loss function for the first stage is replicating the ultimate loss function across time and frequency:
$$
\mathcal{L}_{I}\left(f_{I, \theta}(x), y\right)=\sum_{t, k} \mathcal{L}\left(f_{I, \theta}(x)[t, k], y\right) \tag{6}
$$
$f_{I, \theta}(x)[t, k]$ is the output of the first part of the network at time $t$ and frequency $k$. The pre-trained parameters are then used to initialize the network for the second stage where the whole network is trained with the loss function Eq. 5.

Previous experiments have shown that the two-stage training is necessary, as the network is deep and directly training it from scratch is prone to local optima.

# Domain Adaptation

## A. Supervised Adaptation

The idea of domain adaptation is to train a model using both simulated (source domain) and real (target domain) data so that the model has the best performance in real test scenarios.

To apply supervised domain adaptation,

1. Use the simulated data to pre-train a model, which is the initialization of the subsequent optimization processes. 
2. Then train a model that minimizes the loss on both the source domain and the target domain:

$$
\theta^{*}=\underset{\theta}{\arg \min } \mu_{t} \underset{(x, y) \in D_{t}}{\mathbf{E}} \mathcal{L}\left(f_{\theta}(x), y\right)+\mu_{s} \underset{(x, y) \in D_{s}}{\mathbf{E}} \mathcal{L}\left(f_{\theta}(x), y\right) \tag 7
$$

$\mu_t$: weighting parameters for the loss on the target domain 

$\mu_s$: weighting parameters for the loss on the source domain  

$D_t$: a set of fully-labeled real audio data

$D_s$: a set of fully-labeled simulated audio data

In practice, the weighting is implemented by changing the proportion of source and target domain samples in each mini-batch. (这个权重是由每个 batch 中 source domain 和 target domain 的样本数比例决定)

## B. Weakly-Supervised Adaptation

Annotation of the real samples requires a heavy workload.

Therefore propose a weakly-supervised adaptation scheme to further reduce the effort for data collection.

$D_w$: a set of weakly-labeled real audio data 弱标注是指仅给出声源数

$D_s$: a set of fully-labeled simulated audio data

Each value $z_i$ from the weak label domain $Z$ indicates the number of sources in the audio frame $x_i$.

We apply the adaptation by minimizing a weak supervision loss $\mathcal{L_w}$ on the target domain as well as the supervised loss (Eq. 5) on the source domain:
$$
\theta^{*}=\underset{\theta}{\arg \min } \mu_{w} \underset{(x, y) \in D_{w}}{\mathbf{E}} \mathcal{L_w}\left(f_{\theta}(x), z\right)+\mu_{s} \underset{(x, y) \in D_{s}}{\mathbf{E}} \mathcal{L}\left(f_{\theta}(x), y\right) \tag{8}
$$
Define the weak supervision loss as the minimum distance in the output space between the network output and all possible labels that satisfy the weak label:
$$
\mathcal{L}_{w}\left(f_{\theta}(x), z\right)=\min _{y \in r(z)}\left\|f_{\theta}(x)-o^{*}(y)\right\|_{2}^{2} \tag 9
$$
$r(z)$ is the set of all sound DOA labels that satisfy the weak label $z$, i.e. the number of sources in $y$ is $z$:
$$
r(z) = \{y: |y|=z\}
$$
![image-20211026204007421](https://tva1.sinaimg.cn/large/008i3skNly1gvszvmq73vj30ev0mamyv.jpg)

The effectiveness of the weakly-supervised adaptation depends on the initial performance of the network model. If the network initial output is too far away from the ground truth, the weak supervision will lead to incorrect pseudo-labels.

## C. Pseudo-labeling with Data Augmentation

![image-20211026203843156](https://tva1.sinaimg.cn/large/008i3skNly1gvszu777jej30jw0lymz2.jpg)

如 Fig. 5b & Fig. 5e 所示，多声源定位很容易预测错，所以为了让伪标签尽量接近真实标签，对单声源进行伪标注而不是多声源。



**Data augmentation**
$$
D_a = \{(x_i,\mathbf{u}_i)\}_{i=1}^{N_a}
$$
$D_a$: augmented mixture dataset

a set of mixture $x_i$ and their single-source components $\mathbf{u}_{i}=\left\{u_{i j}\right\}_{j=1}^{z_{i}}$

$\{u_{ij}\}$: single-source segments randomly sampled from the weakly-labeled dataset $D_w$, $z_i$ is the number of sources



**Pseudo-labeling on components**

First apply pseudo-labeling (Eq. 11) to its single-source components
$$
p_{\theta}(x, z)=\underset{y \in r(z)}{\arg \min }\left\|f_{\theta}(x)-o^{*}(y)\right\|_{2}^{2} \tag{11}
$$
![image-20211026210044244](https://tva1.sinaimg.cn/large/008i3skNly1gvt0h2sv7pj30k20mmtbx.jpg)

# Experiment

## A. Microphone Array and Data

**Microphone array**

2 versions of the robots: $P1$ and $P2$ differ in their microphone directivity patterns: directional and omni-directional (定向和全向的 / 心形指向和全指向)

**Source-domain data**

Generated the source domain data by convolving clean speech audio with RIRs.

![image-20211026210700704](https://tva1.sinaimg.cn/large/008i3skNly1gvt0nlxcxrj30f70h0400.jpg)

Both the microphone array and the sound source were randomly placed in the room. The distances between the microphone array, the sound source and the walls were at least 0.5 m.

**Target-domain data** (Real data: SSLR)

During each piece of recording the sound source locations are fixed, therefore the coverage in terms of source locations in the real recordings is considerably less than that of the simulated data.

## B. Training Parameters

Pretrain: one epoch in the first stage (Eq. 6) and four epochs on the second stage (Eq. 5).

Then the pretrained model was used as the initial model for the weakly-supervised domain adaptation.

We controlled the weights of the components in the optimization target Eq. 15 to be $μ_w$ = 0.9, $μ_a$ = 0.1, and $μ_s$ = 1.0. This is equivalent as composing mini-batches using 45%, 5% and 50% of the samples from the weakly-labeled dataset, augmented dataset, and the simulated dataset, respectively.

lr: 0.001 and reduced it by half once the training loss no longer decreased

Adam optimizer

mini-batch size: 100

## C. Analysis of Pseudo-Labeling

computed the loss gain between the MSE loss (Eq. 5) of the model prediction and that of the pseudo-label:
$$
\Delta_{L}=\mathcal{L}\left(f_{\theta}(x), y\right)-\mathcal{L}\left(o^{*}\left(p_{\theta}(x, z)\right), y\right) \tag{16}
$$
A positive loss gain indicates the pseudo-labeling is beneficial for the model.

![image-20211027093635378](https://tva1.sinaimg.cn/large/008i3skNly1gvtmbjjl9vj30hi0b9jsl.jpg)

The green bars indicate positive gain (correct weak supervision), while the red bars indicate negative gain (incorrect weak supervision).

![image-20211027093455898](https://tva1.sinaimg.cn/large/008i3skNly1gvtm9td5ifj30ho0b3jsl.jpg)

![image-20211027093926630](https://tva1.sinaimg.cn/large/008i3skNly1gvtmei9w5bj30ho0bfdgy.jpg)

Since the modified adaptation relies on pseudo-labels of the single- source components, it generates more reliable results than the direct application of pseudo-labeling on the multi-source frames.



## D. DOA Estimation Evaluation Protocol

The neural network models were trained on fully-labeled simulated data, weakly-labeled (for weakly-supervised approaches) or fully-labeled (for supervised approaches) real data, and augmented data if applicable.

Two evaluation settings: (a) the number of sound sources is known, or (b) unknown

(a) MAE(°) and ACC(%)
$$
\mathrm{MAE}=\frac{\sum_{i} \sum_{j=1}^{z_{i}} d\left(\hat{\phi}_{i j}, \phi_{i j}\right)}{\sum_{i} z_{i}}
$$
![image-20211027094648524](https://tva1.sinaimg.cn/large/008i3skNly1gvtmm64mqlj30yn08dwg4.jpg)

**SUPREAL** fully-supervised approach using only fully-labeled real data for two-stage training.

**SUPSIM** trained with only the simulated data. (This is also the pre-trained model for the domain adaptation approaches.)

**ADSUP** The supervised adapted model, i.e. pre-trained with the simulated data and then adapted using the fully-labeled real data in a supervised fashion (Eq. 7).

**ADWEAK** The weakly-supervised adapted model without using augmented data, i.e. pre-trained with the simulated data and then adapted using the weakly-labeled real data with the minimum distance adaptation scheme (Eq. 8).

**ADPROP** pre-trained with the simulated data and then adapted using the weakly-labeled real data and augmented data with the adaptation scheme (Eq. 15).

z = 2, the performance of ADPROP is significantly better compared to ADWEAK.



(b) Precison and Recall

![image-20211027094839228](https://tva1.sinaimg.cn/large/008i3skNly1gvtmo3isosj30ym0ntgr8.jpg)

## G. Scalability with Data Size

![image-20211027102201349](https://tva1.sinaimg.cn/large/008i3skNly1gvtnmu51w4j30i90efabs.jpg)

how F1-score evolve with the size of the target-domain training data.

# Conclusion

We have proposed a framework to train deep neural networks for multi-source DOA estimation. The framework uses simulated data together with weakly labeled data under a domain adaptation setting. We have also proposed a data augmentation scheme combining our weakly-supervised adaptation approach with reliable pseudo-labeling of mixture components in the augmented data. This approach prevents incorrect adaptation caused by difficult multi-source samples. The proposed weakly-supervised method achieves similar per- formance as the fully-labeled case under certain conditions. 

Overall, the proposed framework can be used for deploying learning-based sound source localization approaches to new microphone arrays with a minimal effort for data collection.

# You may also enjoy:
Frank Jagger：The Cone of Silence-Speech Separation by Localization阅读笔记
Frank Jagger：TDOA estimation using DNN with TF Mask阅读笔记
Frank Jagger：DNN for Multiple Speaker Detection and Localization阅读笔记
Frank Jagger：Neural Network Adaptation and Data Augmentation for Multi-Speaker DOA Estimation阅读笔记
Frank Jagger：Broadband DOA Estimation using CNN trained with noise signals阅读笔记
Frank Jagger：Multi-speaker DOA estimation using deep CNN trained with noise signals阅读笔记
Frank Jagger：SLoClas: A DATABASE FOR JOINT SOUND LOCALIZATION AND CLASSIFICATION阅读笔记
Frank Jagger：DOA estimation for multiple sound sources using CRNN阅读笔记
Frank Jagger：Robust Source Counting and DOA Estimation Using SPS and CNN阅读笔记
Frank Jagger：Sound Event Localization and Detection of Overlapping Sources Using CRNN阅读笔记
Frank Jagger：Recursive speech separation for unknown number of speakers阅读笔记
Frank Jagger：CRNN-Based Multiple DoA Estimation Using Acoustic Intensity Features for Ambisonics Recordings阅读笔记
Frank Jagger：End-to-end Binaural Sound Localisation from the Raw Waveform阅读笔记
