# Distortion-controlled training for end-to-end reverberant speech separation with auxiliary autoencoding loss 阅读笔记

# Abstract
Performance of speech enhancement and separation systems in reverberant environments is yet to be explored. A core problem in reverberant speech separation is about the training and evaluation metrics. Standard time-domain metrics may introduce unexpected distortions during training and fail to properly evaluate the separation performance due to the presence of the reverberation.

We first introduce the "equal-valued contour" problem in reverberant separation where multiple outputs can lead to the same performance measured by the common metrics. We then investigate how "better" outputs with lowest target-specific distortions can be selected by auxiliary autoencoding training (A2T). A2T assumes that the separation is done by a linear operation on the mixture signal, and it adds an loss term on the autoencoding of the direct-path target signals to ensure that the distortion introduced on the direct-path signals is controlled during separation. Evaluations on separation signal quality and speech recognition accuracy show that A2T is able to control the distortion on the direct-path signals and improve the recognition accuracy.

# 1. Intro
