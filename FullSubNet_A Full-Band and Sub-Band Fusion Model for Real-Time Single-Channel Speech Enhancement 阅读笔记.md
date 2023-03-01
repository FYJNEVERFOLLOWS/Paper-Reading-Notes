#! https://zhuanlan.zhihu.com/p/610174244
# FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Speech Enhancement 阅读笔记

# FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement 阅读笔记

# Abstract
Full-band and sub-band refer to the models that input full-band and sub-band noisy spectral feature, output full-band and sub-band speech target, respectively. The sub-band model process each freq independently, with input of one freq and several context freqs, output is the prediction of the clean target for the corresponding freq. 

Full-band model can capture the global spectral context and the long-distance cross-band dependencies. It lacks the ability to model signal stationarity and attend the local spectral pattern.
Sub-band is just the opposite.

FullSubNet connect a pure full-band model and a pure sub-band model sequentially and use practical joint training to integrate these two types of models' advantages.

Exp results show that full-band and sub-band info are complementary, and FullSubNet can effectively integrate them.

# Intro
Due to the high dimension and the lack of apparent geometric structure for the time domain signal, the freq domain methods still dominate the vast majority of speech enhancement methods.

Sub-band based methods are designed on the following grounds
1. learns the freq-wise signal stationarity to discriminate between speech and stationary noise. STFT mag reflects the stationarity, which is the foundation for the conventional noise power estimators and SE methods.
2. focuses on the local spectral pattern presented in the current and context freqs. The local spectral pattern has been proved to be informative for discriminating between speech and other signals.

# Method
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202302/20230209230003.png)

For sub-band model, set $N$ neighbor freqs for each side of the input freq

## 2.2. Learning target
cIRM

# 5. Conclusion
FullSubNet can capture the global (full-band) spectral info and the long-distance cross-band dependencies, meanwhile retaining the ability to modeling signal stationarity and attending the local spectral pattern.