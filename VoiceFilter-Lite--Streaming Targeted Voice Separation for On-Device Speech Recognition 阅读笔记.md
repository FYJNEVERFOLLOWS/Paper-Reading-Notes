# VoiceFilter-Lite: Streaming Targeted Voice Separation for On-Device Speech Recognition 阅读笔记

# Abstract

Spk-conditioned separation system: SpeakerBeam, VoiceFilter

Integrating a spk-conditioned separation model to ASR should:
1. improve the ASR performance when there are multiple voices
2. be harmless to the recognition performance under other scenarios, e.g. when the speech is only from the target speaker, or when there is non-speech noise such as music present in the background.

For streaming systems, to have minimal latency, bi-directional recurrent layers or temporal convolutional layers shall not be used in the model.

VoiceFilter-Lite operates as a frame-by-frame frontend signal processor to enhance the features consumed by the speech recognizer, w/o reconstructing audio signals from the features.

Contributions:
1. Perform separation directly on ASR input features
2. An asymmetric loss func to penalize over-suppression during training, to make the model harmless under various acoustic environments
3. An adaptive suppression strength mechanism to adapt to diff noise conditions

# 2. Review of the VoiceFilter system

At inference time, VoiceFilter takes two audio as input: noisy audio and ref audio from the target spk. A pre-trained spk encoder is used to produce the spk-discriminative embedding (d-vector) from the ref audio. STFT first goes through conv layers, then is frame-wise concatenated with the d-vector, finally goes through LSTM and FC to predict a TF soft mask. 

At training time, VoiceFilter is trained by minimizing a loss func which measures the difference between the clean spectrogram and the masked spectrogram.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202302/20230213223651.png)
# 3. VoiceFilter-Lite