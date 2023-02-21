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
## 3.1. Integration with ASR
Focuses on improving ASR performance, thus unnecessary to reconstruct any audio waveform from the masked spectrogram via iSTFT. Instead, VoiceFilter-Lite operates as a frame-by-frame frontend signal processor that directly takes acoustic features as input, and outputs enhanced acoustic features.

## 3.2. Model topology
Since VoiceFilter-Lite is designed for streaming ASR, we have made several changes to guarantee minimal latency: (1) limit the conv layers to be 1D instead of 2D, meaning the conv kernels are for the freq dimension only; (2) uni-directional LSTM.

To make our VoiceFilter-Lite model robust to various noise conditions, we add more variations to the noisification process: (1) the interference audio sources can be either speech from other spks, or non-speech noise such as ambient noise or background music; (2) the noises can be applied through either additive or reverberant operations; (3) SNR is random within 1 dB to 10 dB.

## 3.3. Asymmetric loss
If we add a voice filtering component to an existing ASR system, we must guarantee that the ASR performance does not degrade for any noise condition, which is a very challenging task. We observed that most of the degradation in WER is from false deletions, which indicates significant **over-suppression** by the voice filtering model.

To overcome the over-suppression problem, we propose a new loss function for masking-based speech separation or enhancement, named **asymmetric loss**. 

Conventional L2 loss on T-F spectrogram related features:
$$
L=\sum_{t}\sum_{f}\left(S_{cln}(t,f)-S_{enh}(t,f)\right)^2
$$
We would like to be more tolerant of under-suppression errors, and less tolerant of over-suppresion errors. Thus define an asymmetric penalty function $g_{\mathrm{asym}}$ with penalty factor $\alpha \gt 1$:
$$
g_{\text {asym }}(x, \alpha)= \begin{cases}x & \text { if } x \leqslant 0 ; \\ \alpha \cdot x & \text { if } x>0 .\end{cases}
$$
Then the asymmetric L2 loss function can be defined as:
$$
L_{\text {asym }}=\sum_t \sum_f\left(g_{\text {asym }}\left(S_{\text {cln }}(t, f)-S_{\mathrm{enh}}(t, f), \alpha\right)\right)^2 .
$$

## 3.4. Adaptive suppression strength
While our model significantly improves ASR performance under speech noise, it can still degrade ASR performance under non-speech noise.

One way to mitigate the performance degradation is to have an additional compensation to the over-suppression at inference time.
$$
S_{\mathrm{out}}^{(t)}=w\ \cdot\ S_{\mathrm{enh}}^{(t)} + (1 - w)\ \cdot\ S_{\mathrm{in}}^{(t)}
$$
Here $w$ is the suppression strength. When $w=0$, voice filtering is completely disabled; when $w=1$, there is no compensation.

In practice, we wish to use a larger $w$ when voice filtering improves ASR, and a smaller $w$ when it hurts ASR. Thus add a second binary classification output to the model, which predicts whether a feature frame is from overlapped speech (class label 1) or not (class label 0). We denote the noise type prediction as
$f_{\text {adapt }}\left(S_{\text {in }}^{(t)}\right)\in [0,1]$, the adaptive suppression strength at time $t$ can be defined as
$$
w^{(t)}=\beta \cdot w^{(t-1)}+(1-\beta) \cdot\left(a \cdot f_{\text {adapt }}\left(S_{\text {in }}^{(t)}\right)+b\right),
$$
where $a>0$ and $b \geqslant 0$ define a linear transform, and $0 \leqslant \beta<$ 1 is a moving average coefficient to make the suppression more smooth.

# 4. Exp
Results meet our goal of having an **always harmless and sometimes helpful** model that is safe to use in real applications.

# 5. Conclusions
In this paper, we described VoiceFilter-Lite, a tiny and fast model that performs targeted voice separation in a streaming fashion, as part of an on-device ASR system. Since modern ASR models are already trained with a diverse range of noise conditions, we need to guarantee that the voice separation model does not hurt the ASR performance, especially under non-speech noise, and reverberant room conditions. We achieved this by training our model with an asymmetric loss function, and applying an adaptive suppression strength at runtime. Combining these novel efforts, we developed a 2.2 MB model that has no WER degradation on the clean and non-speech noise conditions we measured, while largely improving
ASR performance on overlapped speech.