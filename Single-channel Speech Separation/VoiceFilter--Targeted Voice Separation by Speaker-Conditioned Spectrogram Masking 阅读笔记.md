# VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking 阅读笔记

# Abstract
Separates the voice of a target spk from multi-spk signals, by making use of a ref signal from the target spk. We achieve this by training two separate neural networks: (1) A spk recognition network that produces spk-discriminative embeddings; (2) A spectrogram masking network that takes both noisy spectrogram and spk embedding as input, and produces a mask.

# Intro
Spk-dependent speech separation = voice filtering = spk extraction

We first train a LSTM-based spk encoder to compute robust spk embedding vectors, then train separately a T-F mask-based system that takes two inputs: (1) the embedding vector of the target spk, previously computed with the spk encoder; (2) the noisy multi-spk audio. This system is trained to remove the interfering spks and output only the voice of the target spk. This approach can be easily extended to more than one spk of interest by repeating the process in turns, for the ref recording of each target spk.

# 2. Approach
## 2.1. Spk encoder
A 3-layer LSTM network produce a spk embedding (d-vector), taking log-mel filterbank energies as inputs.

## 2.2. VoiceFilter system
Two inputs: d-vector of the target spk, a mag spectrogram computed from a noisy audio.

d-vector is **repeatedly concatenated** to the output of the last layer in every time frame.

# 3. Exp setup
## 3.2. Evaluation

WER: we want to reduce the WER in multi-spk scenarios, while preserving the same WER in single-spk scenarios.

SDR

# 5. Conclusions
To improve:
1. Larger dataset for training spk encoder
2. adding more interfering spks
3. computing d-vectors over several utts instead of only one to obtain more robust spk embeddings


