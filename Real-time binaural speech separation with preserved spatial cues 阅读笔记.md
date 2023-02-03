# Real-time binaural speech separation with preserved spatial cues 阅读笔记

# Intro
Complex-valued filters are applied to all available microphone signals simultaneously to generate binaural outputs with an additional constraint on interaural cues preservation.

Use two beamformers at the same time to generate left and right outputs respectively.

MIMO TasNet can perform listener-independent speech separation across a wide range of speaker angles and preserve both ITD and ILD features with significantly higher quality than the single-channel baseline.


*mask-and-sum* mechanism


## 3.2. Evaluation metrics
SNRi ()

ITD and ILD errors

