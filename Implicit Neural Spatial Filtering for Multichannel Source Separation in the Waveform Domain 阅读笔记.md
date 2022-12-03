#! https://zhuanlan.zhihu.com/p/587615771
# Implicit Neural Spatial Filtering for Multichannel Source Separation in the Waveform Domain 阅读笔记
## [Interspeech 2022]
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221120113112.png)
# Abstract
Present a model that can separate moving sound sources based on their broad spatial locations in a dynamic acoustic scene. We divide the scene into two spatial regions containing, respectively, the target and the interfering sound sources. The model is trained end-to-end and performs spatial processing implicitly, without any components based on traditional processing or use of hand-crafted spatial features.

# Intro
Spatial clues captured by left and right ear help us localize sound sources and focus attention to directions of interest. Similarly, multiple microphones can be used to separate sound sources based on their spatial locations. With modern devices
often featuring two to eight microphones, multichannel processing is becoming increasingly important in commercial applications.

Spatial filtering, a.k.a. beamforming, exploits spatial sampling by multiple mics in order to enhance signals coming from a given location in space.

More recently, DNNs have been applied to multichannel processing with a varying degree of integration with traditional beamformers. 

Spatial processing without explicitly using or generating spatial filters: [Channel-attention dense u-net for multichannel speech enhancement, Multichannel speech enhancement without beamforming, Multichannel speech enhancement by raw waveform-mapping using fully convolutional networks, On end-to-end multi-channel time domain speech separation in reverberant environments]

Observe that often there is a certain natural separation between regions containing target and interfering sources. Divide the space into two predefined regions containing, respectively, target and interfering sources.

Observe that in multi-stage processing frameworks, later stages often do not have access to all info from the input. For example, enhancement post-filter that follows a beamformer does not have access to spatial info that's lost after spatial filtering. Intuitively, a joint separation in spatial, spectral and temporal domain has potential to be more efficient.

We train Spatial Demucs on dynamic scenes with moving sources of the same type; and since the only discriminative feature between target and interfering sources is their presence in one of the two regions, the network has to understand the spatial configuration in order to perform the separation task. **By discriminating between the target and interference sound sources, the network is implicitly performing source localization.
**
Compared against a mask-based MVDR oracle beamformer followed by a single-channel Demucs post-filter. 

# 2. Related work
# 3. Approach
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221127202048.png)

Acoustic transfer functions are then used to steer the beams, i.e., select or build spatial filters for the given locations. In dynamic scenes these estimates need to be constantly updated and, as a consequence, the performance degrades significantly. In this work, specifically consider moving sound sources. In particular, instead of focusing on point-like positions in space, we divide the space into two pre-defined regions containing, respectively, target and interfering sources, with no constraints on source movement inside these regions (i.e., regions are fixed but the sources are free to move). The model is trained to preserve sounds coming from the target region while suppressing sources in the interference region. 

## 3.1. Model architecture
Based on the causal Demucs, a multi-layer convolutional encoder-decoder with U-Net skip connections and a LSTM for long range dependencies, that works directly in the waveform domain.

Unlike most other approaches that use multi-channel data to produce a single-channel output, we ask the network to output target signals at each channel. By doing so, we preserve the spatial info of the target sources for subsequent downstream tasks.

## 3.2. Training
We train the network to perform a task that requires spatial understanding of the acoustic scene. Train the model by minimizing the L1 loss on the raw waveforms for all mic channels.

## 3.3. Dataset
Segments of 3 seconds.

Classify each segment as either target or interference depending on the loudspeaker position with respect to the given array (we discard ambiguous segments in which the speaker crosses from one region to the other). 

Consider two space subdivisions: left-right split and near-far split with 0.7 m being the boundary between near and far field. 

# 4. Exp
## 4.1. Setup
Oracle MVDR has access to ground truth ideal ratio masks used to compute spatial covariances.

**Evaluation metrics:** SI-SDR, the Mel $\ell_2$ loss, i.e., the $\ell_2$ distance of ground truth and predicted signal in Mel spectral domain.

**Spatial subdivision:** 
(1) Two half-spaces – target sources are on the right and interfering sources on the left;
(2) Near/far split – target sources are in far-field and interfering sources in near-field of the array, with 0.7 m as the boundary (note that we choose far sources as targets since this is the more challenging scenario).

## 4.2. Results
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221127210720.png)

oracle MVDR + 1 ch Demucs is the oracle MVDR with single-channel Demucs as post-processing

Results for different number of sources in the two regions are shown in Table 2.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221128200552.png)

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221128200812.png)

# 5. Conclusions
We presented a method for training a multichannel neural model to perform spatial source separation. The approach divides the space into two regions containing, respectively, target and interfering sources. This spatial subdivision is fixed for a given network, however, we can envision using global conditioning in order to enable switching between several predefined configurations. 
