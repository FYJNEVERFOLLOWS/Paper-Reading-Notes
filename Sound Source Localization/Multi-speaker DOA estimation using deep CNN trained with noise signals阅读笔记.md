# Multi-speaker DOA estimation using deep CNN trained with noise signals阅读笔记
# Abstract

Multi-speaker DOA estimation is formulated as a multi-class multi-label classification probelm, where the assignment of each DOA label to the input feature is treated as a separate binary classification problem.

The phase component of the STFT coefficients of the received microphone signals are directly fed into the CNN.

Utilizing the assumption of disjoint speaker activity in the STFT domain, a novel method is proposed to train the CNN with synthesized noise signals.

With an array of M microphones our proposed framework yields the best localization performance with M-1 convolution layers. (M-1 convolution layers are required to learn from the phase correlation between all the microphone pairs.) 

# Introduction

The relative direction of a sound source with respect to a microphone array is generally given in terms of the direction of arrival(DOA) of the sound wave originating from the source position.

Two kinds of DOA estimation paradigms: broadband and narrowband DOA estimation.

In narrowband DOA estimation, the task of DOA estimation is performed separately for each frequency sub-band, whereas in broadband DOA estimation the task is performed for the whole input spectrum. (The focus is on BB DOA estimation in this work.)

Compared to the signal processing based approaches, supervised learning approaches, being data driven, have the advantage that they can be adapted to different acoustic conditions via training.

Explicit feature extraction steps (GCC, eigenvectors) generally lead to a high computational cost.

One of the main reasons for the success of deep learning has been the encapsulation of the feature extraction step into the learning framework.

By studying the traditional signal processing based methods for DOA estimation, it can be seen that most methods exploit the phase difference information between the microphone signals to perform localization.

In [BB DOA Estimation using CNN], rather than involving an explicit feature extraction step, the phase component of STFT coefficients of the input signal were directly provided as input to the neural network.

In [Multi-speaker localization using CNN trained with noise], the previously proposed framework was extended to estimate multiple speaker DOAs. There, a novel method was developed to generate the training data using synthesized noise signals for multi-speaker localization. One of the main challenges of using noise signals for the multi-speaker case is that, for overlapping signals, the phase of the STFT coefficients get combined non-linearly, and depend on the magnitude of the individual signals. This makes
the learning procedure for the CNN difficult. To overcome this problem, the property of W-disjoint orthogonality, which holds approximately for speech signals, was utilized.

In this paper, we further extend the initial work on DOA estimation of multiple speakers presented in [Multi-speaker localization using CNN trained with noise]. First, estimate the posterior probabilities of the active source DOAs at the frame-level. Then, these frame-level probabilities are averaged over multiple time frames depending on the chosen block length over which the final DOA estimates are to be obtained. From these averaged posterior probabilities, assuming the number of speakers, $L$, within that block is known, the DOAs corresponding to the classes with the $L$ highest probabilities are chosen as the final DOA estimates.

Even when the CNN is trained to estimate maximum two DOA classes per STFT time frame, at a block level the proposed method can be used to localize greater than two speakers also.

# Problem Formulation

multi-class multi-label classification

![image-20211013112303986](https://tva1.sinaimg.cn/large/008i3skNly1gvdiq36tnwj60ku0fywfc02.jpg)

Training dataset consists of pairs of fixed dimension feature vectors for each STFT time frame and the corresponding true DOA class labels.

In the test phase, given the input feature representation corresponding to a single STFT time frame, the first task is to estimate the posterior probability of each DOA class. Following this, depending on the chosen block length, the frame-level probabilities are averaged over all the time frames in the block. Finally, considering $L$ sources, the DOA estimates are given by selecting the $L$ DOA classes with the highest probabilities.

In this work, we consider the number of sources $L$ to be known. (As an alternative, the number of active sources can be estimated based on the number of clear peaks in the averaged posterior probabilities for a signal block. Also, the recorded signal from a reference microphone can also be used for speaker count estimation using the method proposed in [Classification vs. regression in supervised learning for single channel speaker count estimation])

PS: The DOA estimation in this work is performed for signal blocks that consist of multiple time frames of the STFT representation of the observed signals. (The block length can be chosen depending on the application scenario. For example, for dynamic sound scenes it might be preferable to choose shorter block lengths compared to a scenario when it is known that the sources would be static.)

We assume an independent source DOA model, i.e., the spatial location of the sources are independent of each other. Due to this assumption, multi-label classification can be tackled using the binary relevance method, where the assignment of each DOA class label to the input is treated as a separate binary classification problem.

# Input Representation

We user the phase map as the input feature representation in this work.



The received microphone signals are transformed to the STFT domain using an $N_f$ point discrete Fourier transform (DFT). In the STFT domain, the observed signals at each TF instance are represented by complex numbers. Therefore, the observed signal can be expressed as

$$
Y_{m}(n, k)=A_{m}(n, k) e^{j \phi_{m}(n, k)}
$$
where $A_m(n,k)$ represents the magnitude component and $\phi_m(n,k)$ denotes the phase component of the STFT coefficient of the received signal at the m-th microphone for the n-th time frame and k-th frequency bin. In this work, we directly provide the phase component of the STFT coefficients of the received signals as input to our system.

The input feature for the n-th time frame is formed by arranging $\phi_m(n,k)$ for each time-frequency bin $(n,k)$ and each mic $m$ into a matrix of size $M \times K$, where $K = N_f/2 + 1$ is the total number of frequency bins at each time frame (Nyquist frequency). We call this feature representation as the *phase map*. For example, if we consider a microphone array with $M=4$ mics and $N_f=256$, then the input feature matrix is of size $4 \times 129$.

Given the input representations, the next task is to estimate the posterior probabilities of the $I$ DOA classes for each time frame.

# DOA Estimation with CNNs

The main motivation behind using CNNs is to learn the discriminative features for DOA estimation from the phase map input by applying small local filters to learn the phase correlations at the different frequency sub-bands.



The phase map for the n-th time frame: $\Phi_n$

The posterior probability generated by the CNN at the output: $p(\theta_i|\Phi_n)$

$\theta_i$ is the DOA corresponding to the i-th class.

![image-20211013153015734](https://tva1.sinaimg.cn/large/008i3skNly1gvdpv8ir80j617i0dw75z02.jpg)

In the convolution layers, small filters of size $2 \times 1$ are applied to learn the phase correlations between neighboring mics at each frequency sub-band separately.

Difference with [BB DOA Estimation using CNN]:

In that paper, square filters of size 2 × 2 were used to learn the features from the neighboring frequency bins also.

Why change the filter size?

In the case of multiple speakers neighboring frequency bins might contain dominant activity from different speakers, therefore in this work we use $2 \times 1$ filters.



These learned features for each sub-band are then aggregated by the fully connected layers for the classification task.

At most $M - 1$ convolution layers.

We posit that by using small filters of size 2 × 1, with each subsequent convolution layer after the first one, for each sub-band, the phase correlation information from different microphone pairs are aggregated due to the growing receptive field of the filters, and to learn from the correlation between all microphone pairs, $M − 1$ convolution layers would be required to incorporate this information into the learned features. (需要学习 M - 1 次麦克风相位相关信息)

We utilize the binary relevance method [Classifier chains for multi-label classification] to tackle the multi-label classification problem, therefore the output layer of the CNN consists of $I$ sigmoid units, each corresponding to a DOA class.

Loss function: binary cross-entropy

Here, the task of multi-source DOA estimation is performed for a signal block consisting of $N$ time frames. The block-level posterior probability is obtained by averaging $N$ frame-level posterior probabilities for each $\theta_i$, given by
$$
p_{n}\left(\theta_{i}\right)=\frac{1}{N} \sum_{n}^{n+N-1} p\left(\theta_{i} \mid \boldsymbol{\Phi}_{n}\right)
$$
The $L$ DOAs corresponding to the $L$ classes with the highest probabilities are selected as the DOA estimates.

Using more advanced post-processing methods, such as automatic peak detection [Introduction to Algorithms], is beyond the scope of this paper.

# Training Data Generation

Frame-level, so we require an extremely accurate VAD method in order to avoid including silent time frames in the training data, and errors in this task can adversely affect the training.

To avoid this problem, in [BB DOA Estimation using CNN], we proposed to use synthesized noise signals to generate the training data for the single speaker scenario.

To effectively use synthesized noise signals to generate the training data, and taking into account the aim to localize speech sources, we utilize the assumption that the TF representation of two simultaneously active speech sources do not overlap (W-disjoint orthogonality).

Procedure for generating the training data for a scenario with two active speakers:

1. generate the training signals for a single speaker case by convolving the RIRs corresponding to different directions for each acoustic condition considered for training with synthesized spectrally white noise signals.
2. For a specific source array setup, the STFT representation of two multi-channel training signals, corresponding to different DOAs, are concatenated along the time frame axis. Following this, for each frequency sub-band separately, the time-frequency bins for all mics are randomized to get a single training signal.
3. The phase map corresponding to each time frame, for all training signals, is extracted to form the complete training dataset.

Two essential things regarding the randomization process:

1. The randomization of the TF bins is done separately for each frequency sub-band, such that the order of the frequency sub-bands remains the same for different time frames. This is essential since phase correlations are frequency dependent and for all the different time frames, preserving the spectral structure can aid the feature learning.
2. For each freq sub-band, the TF bins for all mics are randomized together, such that phase relations between the mics for the individual TF bins are preserved.

![image-20211015161511940](https://tva1.sinaimg.cn/large/008i3skNly1gvg2el3t3gj60lk0cfwfz02.jpg)

12.4 million time frames

Adam gradient-based optimizer

mini-batches of 512 time frames

lr: 0.001

dropout: 0.5

# Experimental Evaluation

ULA with $M=4$ mics with inter-mic distance of 8 cm.

The input signals are transformed to the STFT domain using a DFT length of $N_f = 512$, with 50% overlap, resulting in $K=257$.

sr: 16kHz

To form the classes, we discretize the whole DOA range of a ULA with a 5° resolution to get $I=37$ DOA classes.

A. Baselines and Objective Measures

For the objective evaluation, two different measures were used: MAE and localization accuracy.

The mean absolute error computed between the true and estimated DOAs for each evaluated acoustic condition is given by
$$
\operatorname{MAE}\left(^{\circ}\right)=\frac{1}{L C} \sum_{c=1}^{C} \sum_{l=1}^{L}\left|\theta_{l}^{c}-\widehat{\theta}_{l}^{c}\right|
$$
L: the number of simultaneously active speakers

C: the total number of speech mixture segments considered for evalutaion for a specific acoustic condition.

without hat: the true DOAs

with hat: the estimated DOAs for the l-th speaker in the c-th mixture

In our evaluation, the localization of speakers for a speech segment is considered accurate if the distance between the estimated and the true DOA for all the speakers is less than or equal to 5°.

B. Experiments with simulated RIRs



to be continue...



# Conclusion

It was empirically shown that for a microphone array with M microphones, $M-1$ convolution layers are required for the best localization performance. The choice of $M-1$ convolution layers is required for the aggregation of the phase correlation information from all microphone pairs in the extracted features.

