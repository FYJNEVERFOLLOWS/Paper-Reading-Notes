# Dual-Path Filter Network: Speaker-Aware Modeling for Speech Separation 阅读笔记

# Abstract
Propose Dual-Path filter network, which focuses on the post-processing of speech separation to improve speech separation performance. DPFN is composed of two parts: the speaker module (that infers the identities of the speakers) and the separation module (uses the speakers' info to extract the voices of individual speakers from the mixture). Avoids the problem of PIT.

# Introduction
In encoder-separator-decoder framework (TasNet, DPRNN, DPT-Net, SepFormer), the dual-path method is the mainstream of the separator. 

Dual-path processes the waveform from the two dimensions of the local path and the global path.

Limitation: SS models can only deal with recordings of a fixed number of speakers.

Speaker extraction can support SS as we extract one target speaker at a time.

In our model, the speaker module takes the mixed or initially separated waveform as input and provides the speaker identity to the separation module. 

How to incorporate the speaker identity into the separation module? - WaveSplit, Speaker-Conditional Chain Model (SCCM), TasTas.

Contributions:
1) filter-based model that focuses on the post-processing of speech separation
2) provides a brand new speaker module to produce representative speaker filters
3) do not need PIT in the training phase

# DPFN
Cascade DPFN to a pre-trained separation model to filter a cleaner source waveform.
![](https://tva1.sinaimg.cn/large/e6c9d24ely1h39a51gejaj20km0ck75k.jpg)
## 3.1. Model Design
## 3.1.1. Speaker Module for Known Speakers
Use the pre-trained SRE16 x-vector model from Kaldi as the speaker module.

Used the recordings in WSJ0 to obtain the x-vector of each speaker in the dataset.

The x-vector is passed through a FC layer and then used as the speaker condition by the separation module.

## 3.1.2. Speaker Module for Unknown Speakers
When only the mixed waveform is given, we first separate the mixture through a pre-trained separation model and input the separated waveforms into our speaker module to obtain speaker filters.