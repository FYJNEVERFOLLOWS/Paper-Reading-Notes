#! https://zhuanlan.zhihu.com/p/447506072
# CRNN-Based Multiple DoA Estimation Using Acoustic Intensity Features for Ambisonics Recordings阅读笔记
# Introduction

CRNN to estimate the DOAs from a first-order Ambisonics (FOA) recording.

Input: features derived from the acoustic intensity vector

Consider a normalized expression of the acoustic intensity vector in each time-frequency bin and propose to use its coefficients as input features.

# Background

**A. Ambisonics Format**

The sound field is recorded by a spherical microphone array and converted into Ambisonics with an encoding matrix.

FOA corresponds to the coeff of the decomposition in the spherical harmonics of order 0 (channel W) and 1 (channels X, Y and Z)
$$
\left[\begin{array}{c}
W(t, f) \\
X(t, f) \\
Y(t, f) \\
Z(t, f)
\end{array}\right]=\left[\begin{array}{c}
1 \\
\sqrt{3} \cos \theta \cos \phi \\
\sqrt{3} \sin \theta \cos \phi \\
\sqrt{3} \sin \phi
\end{array}\right] p(t, f)
$$
在 $p(t,f)$ 处的 FOA 表示。



**B. Acoustic Intensity**

The active intensity vector $I_a(t,f) = \mathcal{R}\{p(t,f)v^*(t,f)\}$ represents the flow of sound energy in a point of space, with $v(t,f)$ the particle velocity.
$$
\mathbf{v}(t, f)=-\frac{1}{\rho_{0} c \sqrt{3}}\left[\begin{array}{c}
X(t, f) \\
Y(t, f) \\
Z(t, f)
\end{array}\right]
$$
$\rho_0$: the density of air; $p(t,f)=W(t,f)$

The active intensity vector disregarding the constant:
$$
\mathbf{I}_{\mathrm{a}}(t, f)=-\left[\begin{array}{l}
\mathcal{R}\left\{W(t, f) X^{*}(t, f)\right\} \\
\mathcal{R}\left\{W(t, f) Y^{*}(t, f)\right\} \\
\mathcal{R}\left\{W(t, f) Z^{*}(t, f)\right\}
\end{array}\right]
$$
The reactive intensity vector $I_r(t,f) = \mathcal{I}\{p(t,f)v^*(t,f)\}$, represents dissipative local energy transfers.
$$
\mathbf{I}_{\mathrm{r}}(t, f)=-\left[\begin{array}{l}
\mathcal{I}\left\{W(t, f) X^{*}(t, f)\right\} \\
\mathcal{I}\left\{W(t, f) Y^{*}(t, f)\right\} \\
\mathcal{I}\left\{W(t, f) Z^{*}(t, f)\right\}
\end{array}\right]
$$


# DOA Estimation System

**A. Input Features**

Propose to exploit both the active and reactive intensity vectors across all freq bins in the STFT domain as inputs to the neural network in a given time frame. Motivated by the fact that the active intensity relates more directly to the DOA and the reactive intensity indicates whether a given time-freq bin is dominated by direct sound from a single source, as opposed to overlapping sources or reverberation.

normalize the inputs in each tf bin regardless of the sound power:
$$
\frac{-1}{C(t,f)}\left[\begin{array}{c}
\mathbf{I}_{\mathrm{a}}(t, f) \\
\mathbf{I}_{\mathrm{r}}(t, f)
\end{array}\right]
$$
**B. Target Outputs and Training Cost**

The target output of the CRNN is a binary vector of size $n_{DOA} \times 1$, each index corresponds to one discrete DOA. The element of the target vector that is the closest to the true DOA is set to 1. (>1 elements can be set to 1 when multi-sources)

Train a specific neural network for each number of sources.

**C. Network Architecture**

![image-20211214104951690](https://tva1.sinaimg.cn/large/008i3skNly1gxd66mhd9yj30u015ogri.jpg)

T (num of frames): 25, F (num of freq bins): 513, C (num of feature channels): 6

Convolutional modules aim to extract spatial information from the inputs. (Convolve along freq)

The second part (2 BiLSTM and 2 FC) uses this information to estimate the DOAs.

**D. From Framewise to Global DOA Estimation**

# Analysis by LRP

Layer-wise Relevance Propagation (LRP) is a technique for determining which features in a particular input vector contribute most strongly to a neural network’s output. 



# Shared Experimental Settings for DOA Estimation

16kHz

STFT: win1024, hop 512



**B. Training Procedure**

Each network could be used to predict any number of sources, but training each network for a specific number of sources yielded better results.



neighborhood of the peak: $\Delta=2\alpha$, $\alpha$ is the angular resolution.



Nadam optimizer, initial lr 10e-3, 0.2 for the single-source network/0.3 for the two-source network dropout after conv block, FC and on the recurrent weights of the BiLSTM layers



early stopping with a patience of 20 epochs. 80/150 epochs for the single-source network and the two-source network.