#! https://zhuanlan.zhihu.com/p/512083025
# The Cone of Silence- Speech Separation by Localization阅读笔记
# Abstract

Given a multi-microphone recordings of an unknown number of speakers talking concurrently, we simultaneously localize the sources and separate the individual speakers. At the core of our method is a deep network, in the waveform domain, which isolates sources within an angular region $\theta\pm\omega/2$, given an angle of interest $\theta$ and angular window size $\omega$. By exponentially decreasing $\omega$, we can perform a binary search to localize and separate all sources in logarithmic time. Our algorithm allows for an arbitrary number of potentially moving speakers at test time, including more speakers than seen during training. Experiments demonstrate state-of-the-art performance for both source separation and source localization, particularly in high levels of background noise.

# Introduction

The directionally sensitive CoS network can help yield solutions to 1) sound localization, 2) audio source separation.

CoS approach enables true *cancellation* of audio sources outside a specified angular window

重要假设：

* Focus only on azimuth angles (the method could equally be applied to elevation angles), meaning that we assume the sources have roughly the same elevation angle.

* Explore circular microphone arrays.

![image-20210822114835134](https://tva1.sinaimg.cn/large/008i3skNgy1gtpf8h44ywj60yi0i7gps02.jpg)

Regions without sources will produce empty outputs.

# Related work

A recent trend is the use of deep neural networks for multi-source direction of arrival (DOA) estimation.

One key challenge is that the number of speakers in real world scenarios is often unknown or non- constant. Many methods require a priori knowledge about the number of sources.

# Method

$\theta$ 和 $\omega$ 是分别进行训练得到的。

**Problem Formulation**

Given M microphones, the problem of M-channel source separation and localization can be formulated in terms of estimating N sources $s_1,...,s_N$ and their corresponding angular position $\theta_1,...\theta_N$.

Our coordinate system最核心的就是麦克风阵列，而 $\theta_i$ 就是基于该系统定义的。

**Base Architecture**

CoS network is adapted from the Demucs architecture. 改动的部分在于 the number of input and output channels (the number of microphones)

![image-20210822114925122](https://tva1.sinaimg.cn/large/008i3skNgy1gtpf9a9qobj60yv0h70vb02.jpg)

**Target Angle $\theta$**

$\theta$ 的编码形式$x'$，$x'$ 是原信号 $x$ 在每个 channel 上根据 TDOA 平移得到的。（对给定的 $\theta$，TDOA是确定的）
$$
\mathbf{x}_{i}^{\prime}=\operatorname{shift}\left(\mathbf{x}_{i}, T_{\text {delay }}\left(p_{\theta}, \operatorname{mic}_{0}\right)-T_{\text {delay }}\left(p_{\theta}, \operatorname{mic}_{i}\right)\right) \quad i=1, \ldots, M-1
$$

$$
T_{\text {delay }}\left(p_{\theta}, \operatorname{mic}_{i}\right)=\left\lfloor\frac{d\left(p_{\theta}, \operatorname{mic}_{i}\right)}{c} \cdot s r\right\rfloor
$$

其中，c: 声速，sr: 采样率，$p_\theta$: $\theta$ 方向上远场声源的位置。$d(p_\theta,mic_i)$是 $p_\theta$ 和 $mic_i$ 之间的欧氏距离



一个需要注意的假设：远场麦克风阵列的TDOA主要取决于方位角（因为可以视为平面波）

选择$mic_0$作为基准，让 $x'_i$ 平移到和 $x'_0$ 对齐的位置

然后训练网络让它输出与 $x'$ 对齐的声源而忽略掉其他的。

**Angular Window Size $\omega$**

有些声源不是点源，而是有一些宽度，并且可能在移动，所以只有 target angle 是不够的。

所以引入第二个变量$\omega$，作为全局条件参数 (global conditioning parameter)

受 WaveNet 的 global conditioning framework 启发，用编码的 one-hot vector **h** 来表示 $\omega$ 

$$
\begin{aligned}
\text { Encoder }_{k+1}=& \text { GLU }\left(\mathbf { W } _ { \text { encoder } , k , 2 } * \operatorname { ReLU } \left(\mathbf{W}_{\text {encoder }, k, 1} * \text { Encoder }_{k}\right.\right.\\
&\left.\left.+\mathbf{V}_{\text {encoder }, k, 1} \mathbf{h}\right)+\mathbf{V}_{\text {encoder }, k, 2} \mathbf{h}\right)
\end{aligned}
$$

$$
\begin{aligned}
\text { Decoder }_{k-1}=& \operatorname{ReLU}\left(\mathbf { W } _ { \text { decoder } , k , 2 } * ^ { \top } \text { GLU } \left(\mathbf{W}_{\text {decoder }, k, 1} *\left(\text { Encoder }_{k}+\text { Decoder }_{k}\right)\right.\right.\\
&\left.\left.+\mathbf{V}_{\text {decoder }, k, 1} \mathbf{h}\right)+\mathbf{V}_{\text {decoder }, k, 2} \mathbf{h}\right)
\end{aligned}
$$



**Network Training**

目标函数

$$
\mathcal{L}\left(\mathbf{x} ; \mathbf{s}_{1}, \ldots, \mathbf{s}_{N}, \theta_{t}, w\right)=\left\|\tilde{\mathbf{x}}^{\prime}-\sum_{i=1}^{N} \mathbf{s}_{i}^{\prime} \cdot \mathbb{I}\left(\theta_{t}-\frac{w}{2} \leq \theta_{i}<\theta_{t}+\frac{w}{2}\right)\right\|_{1}
$$
$x'$ 是输入的 mixture shift 后的信号，$\widetilde{x'}$ 是网络根据 $x'$ 和 $\omega$ 的输出。

# Experiments

![image-20210822183416384](https://tva1.sinaimg.cn/large/008i3skNgy1gtpqyjviiaj60y40g877i02.jpg)

# Limitations

must reduce the angular resolution (Expand Angular Window Size $\omega$) to support moving sources.

assume the microphone array is rotationally symmetric.

# Demo

https://grail.cs.washington.edu/projects/cone-of-silence/

