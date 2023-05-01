#! https://zhuanlan.zhihu.com/p/611079983
# On End-to-end Multi-channel Time Domain Speech Separation in Reverberant Environments 阅读笔记

## ICASSP 2020
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202303/20230302155723.png)
# Abstract

A FCN structure has been used to directly separate speech from multi-microphone recordings, with no need of conventional spatial feature extraction.

[End-to-End Multi-Channel Speech Separation] uses IPDs as additional input, but the win len of the conv kernel used for extracting spectral feats is much smaller than the STFT win len used for extracting spatial features, causing a mismatch and misalignment problem.

In this work we:
1. first design a time-domain separation system which solves the mismatch and misalignment problem mentioned above by employing a trainable 2-D conv layer to build a spatial encoder for spatial feature extr from pairs of mics signals.
2. investigate the influence of reverberation on the separation task and find that applying dereverberation methods as a pre-processing stage can further improve the system's performance
3. perform both SS and ASR exps

# BG
Multi-channel version of Conv-TasNet [End-to-End Multi-Channel Speech Separation]:
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202303/20230302161610.png)

# Multi-channel E2E Separation
Instead of usnig IPDs, this work aims to directly extracting spatial features from time-domain multi-channel signals with a 2-dimensional convolutional layer. Kernal size of $(M=2,L)$ ($M$ channels) keeps the number of frames of spatial features the same as the spectral representation.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202303/20230302171545.png)

# Exp
Results indicate that combining spectral and spatial signal representations in an end-to-end fashion helps improve speech separation and ASR accuracy. Also, dereverberation pre-processing can yield significant performance improvement. Further research is ongoing to extend this system to a multi-device scenario and to evaluate the separation performance with real data recorded in realistic environments such as, for example, CHiME-5

# Conclusion
We argued that conventional spatial features are not optimal for an end-to-end time domain speech separation system. Using a trainable kernel with a window length matched to that of the spectral encoder can efficiently address the misalignment and mismatch problem, leading to a better multi-channel separation performance.