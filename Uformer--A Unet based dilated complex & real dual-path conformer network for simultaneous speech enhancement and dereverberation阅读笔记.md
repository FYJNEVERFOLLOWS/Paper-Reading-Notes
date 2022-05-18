# [复数域语音增强] Uformer: A Unet based dilated complex & real dual-path conformer network for simultaneous speech enhancement and dereverberation 阅读笔记

# Abstract

Traditional approaches always treat complex spectrum and magnitude separately, ignoring their underlying relationship.

Propose *Uformer*, a Unet based dilated complex & real dual-path conformer network in both complex and magnitude domain for simultaneous speech enhancement and dereverberation. Exploit time attention (TA) and dilated convolution (DC) to leverage local and global contextual information and frequency attention (FA) to model dimensional information.

Hybrid encoder and decoder are adopted to simultaneously model the complex spectrum and magnitude and promote the information interaction between two domains.

# Introduction

For a long time, DNN-based speech enhancement algorithms attempt to only enhance the noisy magnitude using IRM and keep the noisy phase when reconstructing speech waveform.

The original conformer model presents two important modules: self-attention module and convolution module to model the global and local information, respectively.

**Overview of transformer and dual-path methods**

To further achieve stronger contextual modeling ability, the combination of conformer and dual-path method is an instinctive idea.

In the magnitude branch of Uformer, we seek to construct the filtering system which only applies to the magnitude domain. In this branch, most of the noise is expected to be effectively suppressed. By contrast, the complex domain branch is established as a decorating system to compensate for the possible loss of spectral details and phase mismatch. Two branches work collaboratively to facilitate the overall spectrum recovery.

![image-20220517151952741](https://tva1.sinaimg.cn/large/e6c9d24ely1h2bfezp5muj20kg0df3zq.jpg)

# Proposed Method

## 2.1. Problem Formulation

In the time domain:
$$
o(t)=s(t)*h(t)+n(t)=s_e(t)+s_1(t)+n(t)
$$
$o(t)$: the observed signal

$s(t)$: the anechoic speech

$h(t)$: room impulsive response (RIR)

$n(t)$: noise

$*$: convolution operation

$s_e(t)$: direct sound plus early reflections

$s_1(t)$: late reverberation

In the frequency domain:
$$
O(t,f)=S_e(t,f)+S_1(t,f)+N(t,f)
$$
Our target is to estimate the $s_e$ and $S_e$ in this work.

## 2.2. Complex Self Attention

Original self attention:
$$
Q=XW_Q,K=XW_K,V=XW_V
$$
The input $X$ is mapped with different learnable linear transformation $W$ to get queries, keys and values.

Then, the dot product results of queries with keys are computed, followed by division of a constant value $k$, representing the projection dimension of $W$. Applying the softmax function to generate the weights and obtain the weighted result:
$$
Attention(Q,K,V)=softmax(\frac{Q^TK}{\sqrt{k}}V)
$$
Complex self attention:

Given the complex input $X$, the complex valued $Q,K,V$ are calculated by:
$$
Q^\mathcal{R}=X^\mathcal{R}W^\mathcal{R}_Q-X^\mathcal{I}W^\mathcal{I}_Q
\\
Q^\mathcal{I}=X^\mathcal{R}W^\mathcal{I}_Q+X^\mathcal{I}W^\mathcal{R}_Q
$$
The complex self attention is calculated by:
$$
ComplexAttention(Q,K,V)=\\

(Attention(Q^\mathcal{R},K^\mathcal{R},V^\mathcal{R})-Attention(Q^\mathcal{R},K^\mathcal{I},V^\mathcal{I})-\\
Attention(Q^\mathcal{I},K^\mathcal{R},V^\mathcal{I})-Attention(Q^\mathcal{I},K^\mathcal{I},V^\mathcal{R}))+\\
i(Attention(Q^\mathcal{R},K^\mathcal{R},V^\mathcal{I})+Attention(Q^\mathcal{R},K^\mathcal{I},V^\mathcal{R})+
Attention(Q^\mathcal{I},K^\mathcal{R},V^\mathcal{R})-Attention(Q^\mathcal{I},K^\mathcal{I},V^\mathcal{I})).
$$

## 2.3. Dilated Complex Dual-path Conformer

![image-20220517195717390](https://tva1.sinaimg.cn/large/e6c9d24ely1h2bnfpd9mkj20jb0aymyd.jpg)

2 FF layers like a sandwich

According to conformer, we employ half-step residual weights in our FF.



The lower freq bands tend to contain high energies while the higher freq bands tend to contain low energies.

In ConvTasNet, original TCN first projects the input to a higher channel space with Conv1d. Then a dilated depthwise convolution (D-Conv) is applied to get a larger receptive field. The output Conv1d projects the number of channel the same as the input. Residual connection is applied to enforce the network to focus on the missing detail and mitigate gradient disappearance.



## 2.4. Hybrid Encoder and Decoder

Both encoder and decoder model the complex spectrum and magnitude at the same time. $C_i$ and $M_i$ denote the complex spectrum and magnitude output of encoder/decoder layer $i$ respectively. To promote the information exchange between complex spectrum and magnitude, the complex-magnitude fusion results $\hat{C}_i$ and $\hat{M}_i$ are calculated by:
$$
\hat{C}_i^\mathcal{R}=C_i^\mathcal{R}+\sigma(M_i),\\
\hat{C}_i^\mathcal{I}=C_i^\mathcal{I}+\sigma(M_i),\\
\hat{M}_i=M_i+\sigma(\sqrt{{C_i^\mathcal{R}}^2+{C_i^\mathcal{I}}^2}),\\
$$


## 2.5. Encoder Decoder Attention

## 2.6. Loss Function

We multiply the noisy complex spectrum and magnitude with the estimated CRM $H_C$ and IRM $H_R$ respectively to get enhanced and dereverbed complex spectrum and magnitude.

We use hybrid time and frequency domain loss as the target:
$$
\mathcal{L}=\alpha\mathcal{L}_{SI-SNR}+\beta\mathcal{L}_{L1}^T+\gamma\mathcal{L}_{L2}^C+\zeta\mathcal{L}_{L2}^M
$$
SI-SNR loss in time domain, L1 loss in time domain

complex spectrum L2 loss and magnitude L2 loss



# Experiments and Results



# Conclusion

Propose Uformer for simultaneous speech enhancement and dereverberation in both magnitude and complex domains. Leverage local and global contextual info as well as frequency info to improve the speech enhancement and dereverberation ability by dilated complex & real dual-path conformer module. Encoder decoder attention is applied to better model the information interaction between encoder and decoder.

