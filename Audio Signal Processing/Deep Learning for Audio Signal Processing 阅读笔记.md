# [音频信号处理综述] Deep Learning for Audio Signal Processing 阅读笔记

# Abstract

This article provides a review of the SOTA deep learning techniques for audio signal processing.

The dominant feature representations (log-mel spectra and raw waveform) and deep learning models are reviewed.

# Introduction

The recent surge in interest in deep learning has enabled practical applications in many areas of signal processing, often outperforming traditional signal processing on a large scale.

Raw audio samples form a one-dimensional time series signal, which is fundamentally different from two-dimensional images. Audio signals are commonly transformed into two-dimensional time-frequency representations for processing, but the two axes, time and frequency, are not homogeneous as horizontal and vertical axes in an image. 

# Methods

## A. Problem Categorization

![image-20220514163706527](https://tva1.sinaimg.cn/large/e6c9d24ely1h280sfawtpj20js0biab2.jpg)

Single global label - single class: *sequence classification*

Single global label - set of classes: *multi-label sequence classification*

Single global label - numeric value: *sequence regression*

Label per time step - single class: *Sequence labeling*

Label per time step - numeric value: *regression per time step*

*Sequence transduction* (free-length sequence of labels): speech-to-text or language translation.

## B. Audio Features

For decades, MFCC have been used as the dominant acoustic feature representation for audio analysis tasks. These are magnitude spectra projected to a reduced set of frequency bands, converted to logarithmic magnitudes, and approximately whitened and compressed with a discrete cosine transform (DCT). With deep learning models, the latter has been shown to be unnecessary or unwanted, since it removes the information and destroys spatial relations. Omitting it yields the *log-mel spectrum*, a popular feature across audio domains.

The window size for computing spectra trades temporal resolution (short windows) against frequential resolution (long windows). 

## C. Models

**CNNs:** a 1-d temporal convolution or a 2-d time-frequency convolution is commonly adopted in the case of spectral input features whereas a time-domain 1-d convolution is applied for raw waveform inputs.

The receptive field can be increased by using larger kernels or stacking more layers. Especially for raw waveform inputs with a high sample rate, reaching a sufficient receptive field size may result in a large number of param of the CNN and high computational complexity. Alternatively, a dilated convolution can be used. A stack of dilated convolutions enables networks to obtain very large receptive fields with just a few layers, while preserving the input resolution as well as computational efficiency.

**RNNs:** RNNs compute the output for a time step from both the input at that step and their hidden state at the previous step.

Stacking of recurrent layers and sparse recurrent networks have been found useful in audio synthesis.

RNNs can process the output of a CNN, forming a *Convolutoinal Recurrent Neural Network (CRNN)*. In this case, convolutional layers extract local information, and recurrent layers combine it over a longer temporal context.

**Sequence-to-Sequence Models:** fully neural, transduces an input seq into an output seq directly.

The acoustic, pronunciation, and language modeling components are trained jointly in a single ASR system.

One such model is the *connectionist temporal classification* (CTC), which introduces a blank symbol to match the output seq length with the input seq and integrates over all possible ways of inserting blanks to jointly optimize the output seq instead of each individual output label.

The extended CTC model includes a separate recurrent language model component, referred to as the RNN-T (transducer).

Attention-based models learn alignments between the input and output seqs jointly with the target optimization, like LAS.

**GANs:**

GANs consist of two networks, a generator and a discriminator. The generator maps latent vectors drawn from some known prior to samples and the discriminator is tasked with determining if a given sample is real or fake.

**Loss Functions**
The loss function needs to be differentiable with respect to trainable parameters of the system when gradient descent is used for training.

 The MSE between log-mel spectra can be used to quantify the difference between two frames of audio in terms of their spectral envelopes.

 **Phase modeling**
The phase can be estimated from the magnitude spectrum using the Griffin-Lim Algorithm. But the accuracy of the estimated phase is insufficient to yield high quality audio, desired in applications such as in source separation, audio enhancement, or generation. Wavenet is trained to generate a time-domain signal from log-mel spectra.

## D. Data
In image processing, tasks with limited labeled data are solved with *transfer learning*: using large amounts of similar data labeled for another task and adapting the knowledge learned from it to the target domain. Eg, DNNs trained on the ImageNet dataset can be adapted to other classification problems using small amounts of task-specific data by *retraining* the last layers or *finetuning* the weights with a small learning rate.

In ASR, a model can be pretrained on languages with more transcribed data and then adapted to a low-resource language or domains.

*Data generation* and *data augmentation* are other ways of addressing the limited training data problem.

## E. Evaluation
WER for ASR. WER counts the fraction of word errors after aligning the reference and hypothesis word strings and consists of insertion, deletion and substitution rates which are the number of insertions, deletions and substitutions divided by the number of reference words.

Accuracy for acoustic scene classification.

EER (equal error rate) or F-score for event detection.

# Applications
## A. Analysis
### 1) Speech
For decades, the triphone-state GMM / HMM was the dominant choice for modeling speech. These model have their mathematical elegance.

With the adoption of RNNs for speech modeling, the conditional independence assumption of the output targets incurred by the traditional HMM-based phone state modeling is no longer necessary and the research field shifted towards full seq-to-seq models.