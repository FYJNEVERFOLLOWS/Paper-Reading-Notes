#! https://zhuanlan.zhihu.com/p/534733030
# DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement 阅读笔记

# Abstract

Recent studies use complex-valued spectrogram as a training target but train in a real-valued network, predicting the magnitude and phase component or real and imaginary part, respectively. Convolution recurrent network integrates a convolutional encoder-decoder (CED) structure and LSTM, which has been proven to be helpful for complex targets.



Design a new network simulating the complex-valued operation, called DCCRN, where both CNN and RNN structures can handle complex-valued operation.



# Introduction

Focus on DL-based single-channel speech enhancement for real-time processing with low model complexity.

## 1.1. Related Work
Time domain: direct regression (without an explicit signal front-end, involving Conv1d) and adaptive front-end approaches (adopt a Convolution Encoder-Decoder or a U-Net framework, which resembles the STFT and its inversion iSTFT. The enhancement network is then inserted between the encoder and the decoder, like TCN and LSTM, utilizing their capacity of temporal modeling).

TF domain: CED can take complex-valued or real-valued spectrogram as input. 

IBM, IRM and SMM (spectral magnitude mask) ignore the phase information.

DCUNET is trained to estimate CRM and optimizes the SI-SNR loss after transforming the output TF spectrogram to a time-domain waveform by iSTFT.

While achieving SOTA performance with temporal modeling ability, many layers of convolution are adopted to extract important context info, leading to large model size and complexity, which limits its practical use in efficiency-sensitive applications.

## 1.2. Contributions
Design DCCRN optimizing an SI-SNR loss, which combines both the advantages of DCUNET and CRN, using LSTM to model temporal context with significantly reduced trainable parameters and computational cost.

# DCCRN
## 2.1. Convolution recurrent network architecture
The encoder consists of 5 Conv2d blocks aiming at extracting high-level features from the input features, or reducing the resolution. The decoder reconstructs the low-resolution features to the original size of the input, leading the encoder-decoder structure to a symmetric design.

![](https://tva1.sinaimg.cn/large/e6c9d24ely1h3187w1qgqj20ix0823z3.jpg)

## 2.2. Encoder and decoder with complex network
![](https://tva1.sinaimg.cn/large/e6c9d24ely1h318921nc7j20hd0bwmye.jpg)
