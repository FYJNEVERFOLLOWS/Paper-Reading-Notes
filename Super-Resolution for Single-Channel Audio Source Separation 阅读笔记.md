# SFSRNet: Super-Resolution for Single-Channel Audio Source Separation 阅读笔记

# Abstract
Some single-channel source separation methods utilize downsampling to either make the separation process faster or make the NNs bigger and increase accuracy. The problem of downsampling is that the upsampling to reconstruct the audio source estimations in the original sampling rate usually comes with information loss.

Tackle this problem by introducing SFSRNet enclosing a super-resolution (SR) network. Any separation method where the length of the sequence is a bottleneck in speed and memory can be made faster or more accurate by using the SR network.

# Introduction
The audio source separation system takes the given audio mixture as its input and outputs $C$ estimations of the single audio sources.

*estimations*: the estimations of the recovered single audio source

## Basic Idea
Downsampling has the advantages of speeding up and lowering the memory usage of the separation process. However, the issue with downsampling the input signal is that the later required upsampling process to get the original audio signal frequencies are unable to fully restore the information that gets lost during the downsampling process.

Propose a separate super-resolution (SR) network to achieve better upsampling results. Aside from the downsampled estimations, the input audio mixture in its original sampling rate is used as an additional input to improve the upsampling process.

Let us note that our SR network can be added to most existing separation methods to speed them up and make them more accurate.

Since the SR network only consists of a few convolutional layers, it is highly parallelized and its cost can be neglected compared to the overall cost of the separation. In addition, we can increase accuracy by adding more layers to the separation network to improve the separation since the downsampling process does not only save time, but also memory.

## Main Contributions
Propose the SFSRNet approach enclosing a super-resolution network to address the single-channel audio source separation problem. Our approach adopts the downsampling technique of the existing SepFormer architecture.

Main contributions:
1. Improving the existing SepFormer by calculating intermediate estimations after each block to reduce the loss of information
2. Introducing the SR network which can be used to improve most existing separation architectures.

## Related Work


![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202205/20220610145329.png)