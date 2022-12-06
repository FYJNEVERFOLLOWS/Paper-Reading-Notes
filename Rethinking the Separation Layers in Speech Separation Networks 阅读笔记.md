# Rethinking the Separation Layers in Speech Separation Networks 阅读笔记
## ICASSP 2021
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202212/20221202145327.png)
# Abstract

SIMO: single-input-multi-output

SISO: single-input-single-output

Certain two-stage separation systems integrated with a post-enhancement SISO module can improve the separation quality. Why performance improvements can be achieved by incorporating the SISO modules? Are SIMO modules always necessary? We empirically examine those questions by designing models with varying configurations in the SIMO and SISO modules. We show that comparing with the standard SIMO-only design, a mixed SIMO-SISO design with a same model size is able to improve the separation performance especially under low-overlap conditions.

# 1. Intro
Given the mixture signal is usually considered as a single input, separation models can be broadly categorized into SISO systems and SIMO systems. SISO usually consists a stack of one to one mapping layers, extracting one spk from the mixture at each time. SISO is typically designed for guided source separatoin or speech enhancement tasks, where a bias is often needed to distinguish the target speaker.

SIMO systems are the standard design for BSS, which target at separating $C$ sources simultaneously. $C$ masks are generally estimated from the last layer in the network.

A commonly applied integration is to use a SISO network for post-enhancement module on the output of the SIMO separation result, while typically the two modules are not jointly optimized.

Why performance improvements can be achieved by incorporating the SISO modules? For a given model size, how to properly arrange the sizes of SIMO and SISO modules to achieve the best performance? Are SIMO modules always necessary?

In this paper, we empirically analyze different model configs, including the standard SIMO-only model, the mixed SIMO-SISO model (follows the design of a pre-separation and post-enhancement pipeline), and the SISO-only model, on their effectiveness on the separation performance. Unlike the SISO designs in GSS tasks (e.g. spk extraction), the SISO-only models here do not use external bias info but perform separation in an iterative way.

# 2. Configs of SIMO and SISO Modules in Separation Networks
## 2.1. Problem formulation and backbone architecture
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202212/20221202215503.png)

Backbone: DPRNN-TasNet (TasNet equipped with DPRNN)

## 2.2. SIMO-only model design
The output layer of SIMO module is typically a FC layer with $C$ output heads. ($N$ -> $N * C$ in channel dimension)

## 2.3. Mixed SIMO-SISO model design
Each of the intermediate feature $\mathbf{F}_i$ of $C$ intermediate features $\{\mathbf{F}_i\}_{i=1}^C$, together with the encoder output of the mixture signal $\mathcal{E}(\mathbf{y})$ are concatenated and passed to the SISO module, which is shared by all SIMO output features, to generate the final estimations $\{\hat{\mathbf{x}}_i\}_{i=1}^C$. The two modules are jointly optimized and no extra training objective is applied to the intermediate features.

## 2.4. SISO-only model design
The encoder layers are applied only once and the decoder layers are applied in every iteration. In other words, the encoder layers map the mixture into a latent representation shared by all iterations, and the decoder layers separate diff targets based on the representation.

## 2.5. Discussions
In the mixed SIMO-SISO design, both the intermediate feature and the input mixture are sent to the SISO module, which leads to better performance than simply sending the intermediate feature to the SISO module.

Serialized Output Training (SOT) applies the SISO-only config w/o iterative separation [Serialized output training for end-to-end overlapped speech recognition].

# 3. Exp Config

# 4. Results and Discussions
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202212/20221203000304.png)
A deeper design in the SISO module is able to improve the performance. Assigning 70% of the total blocks to the SISO module can be a good config.

Performance obtained by the mixed SIMO-SISO design mainly comes from the low-overlap utts.

As more and more recent models consider data distributions with partially-overlap utterances [Librimix, Continuous speech separation: Dataset and analysis], such mixed SIMO-SISO design should be more practical and beneficial than the standard designs.

![](https://tva1.sinaimg.cn/large/008vxvgGgy1h8rxbzzgs9j30vw0dm76k.jpg)


# 5. Conclusion
Experiment results on various configurations showed that although almost all existing separation systems are SIMO-only, the mixed SIMO-SISO design can improve the separation performance especially on low-overlap utterances. The SISO-only design also achieved slightly better performance than the standard SIMO-only design, challenging the role of the SIMO separation layers in a speech separation system.
