#! https://zhuanlan.zhihu.com/p/433298840
SLoClas: A DATABASE FOR JOINT SOUND LOCALIZATION AND CLASSIFICATION

# Introduction

Sound Localization and Classification (SLC) refers to estimating the spatial location of a sound source and identifying the type of a sound event through a unified framework.

Specifically, a SLC method can be divided into two sub-tasks, Direction-of-Arrival Estimation (DOAE) and Sound Event Classification (SEC).

DOAE approaches: TDOA estimation, maximization of SRP, or acoustic intensity vector analysis.

SEC methods: spectro-temporal, MFCC using GMM, HMM and SVM.

Since in real-world applications, a sound event may transmit from a specific direction, it is reasonable to combine DOAE and SEC with not only estimating the respective associated spatial location, but also detecting the type of sound.

A fairly designed database with different sound event classes recorded from several DOAs would help us to solve many real-world SLC problems.

The SLoClas database is recorded by a 4-channel mic array, including 10 environmental sound classes omitting from 72 different DOAs.

Realistic sound event classes that include sounds corresponding to bells, bottles, buzzer, cymbals, horn, metal, particle, phone, ring and whistle, each having around 100 examples.

To facilitate the study of noise robustness, 6 types of noise signals are recorded at 4 DOAs.

We present a baseline framework on SLC using the proposed SLoClas database that may serve as a reference system for prospective research.

# SLoClas DATABASE

![image-20211028100249003](https://tva1.sinaimg.cn/large/008i3skNly1gvusp6w2z5j30o00fwgmx.jpg)

The microphone array is positioned at the room center and the speaker is shifted by varying the angle from 1°to 360°, at 5° interval from 1.5 metre distance as shown in Figure 1(a). This results in (988 sound classes× 72 DOAs) 71,136 recorded sounds equivalent to 23.27 hours of data.

# REFERENCE SYSTEM ON SLOCLAS DATABASE

SLCnet jointly optimize the SEC and DOAE objectives for sound localization and classification. Uses GCC-PHAT and MFCC as the inputs and produce a DOA label and a sound class label for each audio segment.

![image-20211028141239416](https://tva1.sinaimg.cn/large/008i3skNly1gvuzx3mpclj30kt0odmzr.jpg)

**Embedding extraction**

Concatenate the 6-pair GCC-PHAT features and the 4-channel MFCC features as the network inputs.

Use two stacked Fully Connected Layers, followed by Batch Normalization, ReLU activation, and dropout operation.

**DOA estimation**
$$
p(\theta)=\exp \left(-\frac{|\theta-\dot{\theta}|^{2}}{\sigma^{2}}\right) \tag{2}
$$
Instead of CE loss, we adopt the MSE loss to measure the similarity between $p(\theta)$ and $\hat{p}(\theta)$, formulated as:
$$
\mathcal{L}_{M S E}=\|p(\theta)-\hat{p}(\theta)\|_{2}^{2}
$$

# Experiments

![image-20211029090851365](https://tva1.sinaimg.cn/large/008i3skNly1gvvwratabuj30m00l2gq7.jpg)

## 4.3 Results

![image-20211029090932326](https://tva1.sinaimg.cn/large/008i3skNly1gvvws0jugzj30lq0aedh5.jpg)

# 5. Conclusion

We present the SLoClas database, to support the study and analysis for joint sound localization and classifica- tion. The database contains a total of 23.27 hours of data, including 10 classes of sounds, recorded by a 4-channel mi- crophone array. The sound is played over a loudspeaker from different DoAs, varying from 1◦ to 360◦ at an interval of 5◦. Additionally, to facilitate the study of noise robustness, we record 6 types of outdoor noise at 4 DoAs using the same de- vices. We also propose a reference system (i.e., SLCnet) with the experiments conducted on SLoClas corpus.
