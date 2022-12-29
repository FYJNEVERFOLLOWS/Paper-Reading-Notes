#! https://zhuanlan.zhihu.com/p/595090650
# On the Use of Deep Mask Estimation Module for Neural Source Separation Systems 阅读笔记
## [Interspeech 2022]
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221128220322.png)
# Abstract
Most of the recent neural source separation systems rely on a masking-based pipeline where a set of multiplicative masks are estimated from and applied to a signal representation of the input mixture. The estimation of such masks, in almost all network architectures, is done by a single layer followed by an optional nonlinear activation function.

In this paper, we analyze the role of deeper mask estimation module by connecting it to a recently proposed unsupervised source separation method, and empirically show that the deep mask estimation module is an efficient approximation of the so-called *overseparation-grouping* paradigm with the conventional shallow mask estimation layers.

# 1. Intro
[Rethinking the separation layers in speech separation networks] explored the use of a deeper mask estimation module and observed consistent performance improvement.

The deep mask estimation module, which was referred to as the *SIMO-SISO* configuration in [Rethinking the separation layers in speech separation networks], applied a single-input-multi-output module to generate $C$ features and a deep single-input-single-output module on each of the feature to generate the multiplicative mask.

In this paper, we analyze the role of such deeper mask estimation module in the neural source separation pipelines by connecting it to a recently proposed unsupervised source separation method, the mixture-of-mixtures method (*MixIt*). MixIt performs unsupervised source separation by mixing $K$ mixture signals to form a mixture-of-mixtures (MoMs) which contains $C \ge K$ target sources, and estimates $P \ge C$ outputs from the MoMs as the system outputs. The $P$ outputs are then *grouped* or *summed* into $K$ mixtures that best reconstructs the $K$ input mixtures used to create the MoMs. Since the number of outputs $P$ is always greater or equal to both the number of mixtures $K$ and the number of target sources $C$, such separation configuration can be viewed as an *overseparation* paradigm.

# 2. Deep Mask Estimation Module
## 2.1. System Overview
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202212/20221202144013.png)

In Fig. 1, an encoder first encodes the mixture waveform to a latent representation, and the representation is sent to a separator to estimate $C$ multiplicative masks corresponding to the $C$ target sources, and a decoder reconstructs the waveforms from the $C$ masked mixture representations. Most existing pipelines use a shallow mask estimation module which typically consists of a single FC layer with an optional nonlinear activation function, which is shown in Fig. 1 (A).

Fig. 1 (B) shows the simple modification to the conventional pipeline where multiple stacked layers are used in the mask estimation module. This modification contains the SIMO-SISO config in [Rethinking the separation layers in speech separation networks] which used multiple stacked DPRNN layers in the mask estimation module. Here we simply use a multi-layer perceptron (MLP) with a total of 3 layers with tanh as the nonlinear activation function for the first and second layers. The nonlinear activation function for the last layer is kept the same as the FC layer in the conventional pipeline.

## 2.2. Connection to the Overseparation-grouping Pipeline


![](https://tva1.sinaimg.cn/large/008vxvgGly1h9jly9h0kgj30io0bigmn.jpg)

MAC: https://github.com/Lyken17/pytorch-OpCounter

