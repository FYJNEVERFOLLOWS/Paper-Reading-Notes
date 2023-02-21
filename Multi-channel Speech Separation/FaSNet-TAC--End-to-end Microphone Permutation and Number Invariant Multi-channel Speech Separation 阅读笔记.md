#! https://zhuanlan.zhihu.com/p/566561321
# [FaSNet-TAC] E2E Microphone Permutation and Number Invariant Multi-channel Speech Separation 阅读笔记

# End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation, ICASSP 2020 [[paper](https://ieeexplore.ieee.org/abstract/document/9054177)][[code](https://github.com/yoonsanghyu/FaSNet-TAC-PyTorch)]

# Abstract
It is vital to guarantee the robustness of a system with respect to the locations and numbers of microphones in ad-hoc mic speech separation.

Propose transform-average-concatenate (TAC), a simple design paradigm for channel permutation and number invariant multi-channel speech separation. Based on the filter-and-sum network (FaSNet), a recently proposed end-to-end time-domain beamforming system, we show how TAC significantly improves the separation performance across various numbers of microphones in noisy reverberant separation tasks with ad-hoc arrays.

Moreover, we show that TAC also significantly improves the separation performance with fixed geometry array configuration, further proving the effectiveness of the proposed paradigm in the general problem of multi-microphone speech separation.

# 1. Intro
Deep learning-based beamforming systems, sometimes called *neural beamformers*, have been an active research topic recently. A general pipeline in the design of many recent neural beamformers is to first perform pre-separation on each channel independently, and then apply conventional beamforming techniques such as MVDR or MWF (multi-channel Wiener filtering) based on the pre-separation outputs.

Another pipeline for neural beamformers is to directly estimate the beamforming filters in either time domain or freq domain, which allows for end-to-end estimation of beamforming filters in a fully-trainable fashion. But such systems typically assume knowledge about the number of mics, since a standard network layer can only generate a fix-sized output.

FaSNet directly estimates the time-domain beamforming filters without specifying the number or permutation of the mics. With a two-stage design, the first stage applies pre-separation on a selected reference mic by estimating its beamforming filters, and the second stage estimates the beamforming filters for all remaining mics based on pair-wise cross-channel features between the pre-separation output and each of the remaining microphones. The filters from both stages are convolved with their corresponding channel waveforms and summed together to form the beamformed output. The filter estimation in the second stage is invariant to permutation and number of the mics due to the use of pair-wise features. 

FaSNet suffers from the problem that the performance of the pre-separation stage greatly affects the filter estimation at the second stage, and the use of pair-wise features prevents it from utilizing the information from all microphones to make a global decision during filter estimation. These flaws might cause unstable and unreliable performance especially in ad-hoc array configurations, where the acoustic properties of different microphones’ signals may significantly differ.

Propose *transform-average-concatenate* (TAC), a simple method for mic permutation and number invariant processing that fully utilizes the info from all mics. TAC first *transforms* each channel's feature with a sub-module shared by all channels, and then the outputs are *averaged* as a global-pooling stage and passed to another sub-module for extra nonlinearity. The corresponding output is then *concatenated* with each of the outputs of the first transformation sub-module and passed to a third sub-module for generating channel-dependent outputs.

# 2. Transform-Average-Concatenate Processing
## 2.1. TAC
Consider an $N$-channel mic array with an arbitrary geometry where $N\in\{2,...,N_m\}$. Each channel is represented by a sequential feature $\mathbf{Z}_i, i = 1,...,N$

# Results  
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220917202514.png)

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220917202533.png)
