# Multi-Channel Overlapped Speech Recognition with Location Guided Speech Extraction Network 阅读笔记

# Abstract
Propose a system utilizing spectral, spatial, and angle features of each target speaker for multi-channel far-field overlapped speech recognition.

An iterative update procedure is proposed in which the mask-based beamforming and mask estimation are performed alternatively.

# Introduction
Compared to the close-talk scenario, there are three additional acoustic challenges in far-field speech recognition. Lower SNR; lower direct sound-to-reverberation ratio (DRR); multi-talker overlapped speech.

Beamforming utilizes the spatial information collected from multiple microphones to enhance the target speech, while the NNs learn the regularities in speech magnitude spectra to separate speakers. The separated speech is then passed to the acoustic model for recognition.

As a linear filter, the beamforming has limited spatial discrimination and cancellation power for the interfering audio source, especially when the number of microphones is small.

Propose a simple yet powerful approach for far-field overlapped speech recognition. Different from the blind separation networks such as DC or PIT where the label permutation ambiguity is mainly handled by specially designed objective function, in the proposed framework a location based angle feature is extracted for each speaker in the speech mixture, and then processed to estimate the ratio mask for each target speaker by a uni-directional LSTM.

The proposed system removes the dependency between the number of mixing speakers and network complexity, thus leading to potentially better reconstruction of the target speaker and the generalization in complex acoustic environments.

# Overview of Multi-talker Speech Separation
The main challenge in overlapped speech separation lies in the label permutation problem. When there are multiple speakers talking simultaneously, the separated output have random orders, which causes ambiguity in pairing with the reference and prevents the data-driven method from having correct gradients.

Two families of algorithm: blind speech separation (DC & PIT) and informed speech extraction.

The blind separation system is usually required to estimate the separation for each source simultaneously. Blind separation can be viewed as an "unbiased" separation. The blind separation for each source is usually sub-optimal, due to the equal consideration for all of them simultaneously. And the separation performance drops significantly when more speakers are involved.

In the informed speech extraction systems, an additional source of information was assumed available, which helps to identify each involving speaker and remove the uncertainty in permutation from the input feature perspective.

Helpful clues: speaker identity features extracted from an additional enrollment utterance; vision clue; location based clue.

When an additional clue is provided, during the separation, the network usually has a clear bias toward certain speakers. Therefore this type of speech extraction can be viewed as "biased separation".

# Informed Speech Extraction
## 3.1. Speech extraction network
Utilize the location information as the bias signal, and propose a system to extract the target speaker out of the speech mixture.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202206/20220608151945.png)

The proposed model adopts a similar framework as mask learning systems, where a mask is estimated for the target speaker through a neural network.

To increase the discrimination between speakers, a set of fixed beamformers was applied to pre-process the multi-channel recordings, and the magnitude spectrogram of the beam that points to the target speaker was used to compute the spectral feature.

Intermicrophone Phase Difference is calculated as the spatial feature as follows
$$
\operatorname{IPD}_{i, t f}=\angle\left(\frac{y_{i, t f}}{y_{1, t f}}\right), i=2 \ldots M
$$

where $y$ refers to the observed data in the freq domain, $y_{i,tf}$ is the i-th channel complex spectrum of the mixture signal at time frame $t$ and frequency bin $f$.

The IPD feature captures the relative phase difference between mics, which reflects the TDOA. The IPDs between the first mic and all the other mics are concatenated to be used as the final IPD feature.

Utilize the speaker location to form the bias signal (referred as an angle feature) to help create the target specific feature for later extraction.

To get the angle feature, we first form the steering vector for DOA of each speaker. Then, the cosine distance between the steering vector and the complex spectrum of each channel that is normalized with respect to the first microphone is calculated as follows:

$$
A_{n, t f}=\sum_{i=1}^{M} \frac{e_{n}^{i, f} \frac{y_{i, t f}}{y_{1, t f}}}{\left|e_{n}^{i, f} \frac{y_{i, t f}}{y_{1, t f}}\right|}
$$

$n$: the speaker index

## 3.2. Multi-pass Mask Update
Fixed beamforming has less interference-cancelling power than adaptive beamforming such as MVDR beamformer when the signal statistics are abundant.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202206/20220608171554.png)

## 3.3. Model analysis
2 limitations: 

When the speakers are close, e.g. less than 20 degree, the location bias will not be sufficient to distinguish target speaker;

Total computation is proportional to the number of participants in the mixture, which is more expensive than the blind source separation. This problem could be alleviated by using pruning mechanism based on SSL, i.e. only run the extraction when a sound is detected.