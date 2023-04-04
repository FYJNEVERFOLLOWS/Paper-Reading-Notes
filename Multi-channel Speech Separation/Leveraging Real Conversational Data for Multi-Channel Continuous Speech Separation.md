#! https://zhuanlan.zhihu.com/p/619001521
# Leveraging Real Conversational Data for Multi-Channel Continuous Speech Separation 阅读笔记

## Interspeech 2022
Microsoft Cloud+AI

# Abstract
Existing MC-CSS models are heavily dependent on supervised data - either simulated data which causes data mismatch between training and real-data testing, or the real transcribed overlapping data, which is difficult to be acquired, hindering further improvements in the conversational/meeting transcription tasks.

Propose a 3-stage training scheme for the CSS model that can leverage both supervised data and extra large-scale unsupervised real-world conversational data. The scheme consists of two conventional training approaches—pre-training using simulated data and ASR-loss-based training using transcribed data—and a novel continuous semi-supervised training between the two, in which the CSS model is further trained by using real data based on the teacher-student learning framework.

# 1. Intro
CSS enables streaming processing and is realized with window-based processing, where a speech separation NN model is used to process each windowed signal.

The common practice is to simulate mixed signals from clean speech samples because the ground-truth clean signals are needed for supervised learning, this inevitably causes a mismatch between training and deployment environments. An ASR-loss-based training approach attempts address this issue by using gt transcriptions and an ASR model to define the loss func to minimize. Nonetheless, acquiring accurate transcriptions during speaker-overlapping periods is challenging even for humans, making this approach difficult to scale. In addition, multi-channel recordings may be obtained from different devices with different mic array geometries, creating another challenge to utilize the real conversational data.

Due to the data acquisition challenge mentioned above, it is desirable to leverage real conversational data w/o any supervision signals.

Techniques to leverage unsupervised data for single-channel SE and SS tasks: cycle-consistency-loss-based training [Self-supervised learning for speech enhancement, A Parallel-Data-Free Speech Enhancement Method Using Multi-Objective Learning Cycle-Consistent Generative Adversarial Network], semi-supervised training [Incorporating real-world noisy speech in neural-network-based speech enhancement systems, Remixit: Continual self-training of speech enhancement models via bootstrapped remixing], WavLM, and mixture invariant training.

Propose a 3-stage training framework: (i) pre-training using simulated data, (ii) continuous semi-supervised training using unlabeled real data, and (iii) ASR-loss-based fine-tuning using transcribed real data. This framework allows us to fully leverage both a large amount of unlabeled real data and a small amount of high-quality transcribed real data. Use an array-geometry-agnostic MC model [VarArray].

Unlike existing studies, we use real meeting recordings, i.e., Microsoft internal meetings [22] and the AMI corpus [23], and attempt to obtain insights that are relevant to real application scenarios.

# 2. Background work
## 2.1. VarArray model
It is desirable for a speech separation model to be able to deal with any array geometries without modification or retraining.

VarArray interleaves conformer blocks and cross-channel layers to model both temporal and spatial information of MC signals. It intakes a multi-channel short-time Fourier transform (STFT) sequence and outputs the time-frequency (T-F) masks for two speech sources and two noise sources (i.e., stationary and transient noises).  

## 2.2. Separated signal generation methods
1. Vanilla T-F masking, which directly multiplies the first-channel STFT of the input with the T-F masks. While this method is dominant for single-channel processing, it often causes speech distortion, limiting the benefit of speech separation.
2. MVDR beamforming: MVDR filters are estimated from spatial covariance matrices calculated with the observed STFTs and the T-F masks.
3. ADL-MVDR

# 3. Three-stage CSS training
Real data include large-scale data w/o ref transcriptions nor clean signals, as well as a small amount of transcribed data.

## 3.1. Stage-1: Pre-training with simulated data
$$
\mathcal{L}_{\text {stage-1 }}=\mathcal{L}_{\mathrm{uPIT}}+\sum\limits_{q \in Q} w_q\left\|M_q \odot|Y|-|N|\right\|, \tag{1} 
$$
$$
\mathcal{L}_{\mathrm{uPIT}}=\min \limits_{\phi \in P} \sum \limits_{(i, j) \in P}\left\|M_i \odot|Y|-\left|X_j\right|\right\|, \tag{2}
$$
$P$ possible permutations

$Q$ represents the two types of noises.

$w_q=0.1$ is the weight for the noise losses.

## 3.2. Stage-2: Continuous semi-supervised training
To update the stage-1 model with semi-supervised training, we also generate a teacher model based on the simulated training data, where the teacher model can be the same as the student model or a bigger and more accurate one. Given
the real-recording training samples, the teacher model generates T-F masks. Then, the student model is updated by using these masks as the learning references. The simulated data may also be used together during the stage-2 training.

VarArray has four outputs. Only the first two have two-way permutation ambiguity, while the order of the two noise outputs is fixed.

Stage-2 loss:
$$
\mathcal{L}_{\text {stage-2 }}=\mathcal{L}_{\mathrm{uPIT}}^{\text{T-S}}+\sum\limits_{q \in Q} w_q\left\|M_q^{\text{tea}} \odot|Y|-M_q^{\text{stu}}\odot|Y|\right\|, \tag{3} 
$$
$$
\mathcal{L}_{\mathrm{uPIT}}^{\text{T-S}}=\min \limits_{\phi \in P} \sum \limits_{(i, j) \in P}\left\|M_i^{\text{tea}} \odot|Y|-M_j^{\text{stu}}\odot|Y|\right\|, \tag{4}
$$

Two segmentation schemes for long-form audio signals (longer than 1h): 
1. Fixed-Window Segmentation (FWS), simply splits long-form audio signals into 4-second pieces w/o overlaps. With FWS, some segs may contain >2 spks, which may confuse the model.
2. Conversational Transcription system-based Segmentation (CTS), attempts to control the audio portions to be used for training based on their characteristics. Specifically, the multi-channel
conversational transcription system of [Advances in online audio-visual meeting transcription] is applied to the signals to obtain word-level speaker diarization results. Then, we sweep the long-form signal from the beginning and cut it (1) when the third speaker is detected, (ii) when the length of the current segment exceeds a threshold (20 seconds in our experiments), or (iii) when there is a silence period longer than a threshold (2.5 seconds). This also allows us to sample two-speaker segments more frequently during training to promote speech separation learning.

In addition, to maintain the model’s ability to deal with various microphone arrays and acoustic conditions, a subset of the input channels is randomly chosen for each training sample that is fed to the student model. The stage-2 trained model is expected to be adapted to real data distributions while largely retaining the array-geometry-agnostic property.

## 3.3. Stage-3: ASR-loss-based Fine-tuning
Similar to [VarArray], the VarArray model can be fine-tuned further with an ASR-based loss function, where the VarArray-based mask estimation model, MVDR beamformer, gain adjustment, feature transformation, global mean-variance normalization, and ASR model are combined within a single network.

Only params of VarArray model are updated in this fine-tuning stage, while freezing the other modules.

# 4. Exps
## 4.1. Exp settings
We used the window length and window shift rate of 1.6 seconds and 0.4 seconds, respectively, for CSS in all evaluation systems.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230402171312.png)

## 4.2. Effect of stage-2 continuous semi-supervised training
### 4.2.1. In-domain exps
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230402171714.png)
Comparing the second and third rows shows that the continuous semi-supervised training largely improved the WERs for both non-overlap (Non-Ovlp) and overlap (Ovlp) regions, even when the teacher and student were the same models. This indicates that continuous semi-supervised training forced the CSS model to learn the real data characteristics. Using a larger teacher model in semi-supervised training brought more gains (third and fourth rows) for the student model, which is achieved by the additional effect of knowledge distillation.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230402171910.png)
Continuous semi-supervised training consistently improved the WERs for all the evaluation systems. Better WERs were achieved by the MVDR beamformer and ADL-MVDR, although the relative gains for these systems were smaller than for the masking-based systems. This can be because the beamformers are less sensitive to mask estimation errors.

### 4.2.2. Cross-domain exps
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230402181822.png)
Firstly, by comparing the 1st row
with 2nd and 3rd rows, we observed that stage-2 training al-
ways improved the WER even when the array configurations for
the training data and testing data were different, supporting the
importance of real training data. Secondly, by comparing the 4-
th and 5-th rows, we observed a significant improvement from
stage-1 training, which indicates the necessity of pre-training
using simulated data. Thirdly, as expected, the best WERs of
MS and AMI test sets were achieved by using the matched train-
ing sets as in the 2nd and 3rd rows (6.3% and 5.2% relative
WER reduction for MS and AMI, respectively).

From 3rd and 6th row, using CTS could improve the model performance for overlap regions (32.3% vs. 31.6%). This was because the CTS filtered out most of the silence part in the long-form audios and tended to form more proportions of overlapped segments for training.

DNSMOS is a neural network that predicts the speech quality without requiring the ground-truth speech labels, which was proposed in [DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors].

# 5. Conclusions
In this work, a three-stage training scheme for a multi-channel array-geometry-agnostic CSS model was introduced. Specifically, we proposed to train the CSS model in the order of stage-1 supervised training using simulated data, stage-2 continuous semi-supervised training using real unlabeled data, and stage-3 ASR-loss-based fine-tuning based on real transcribed data.

(ii) the stage-2 training could significantly reduce the train-test mismatch issue that occurred in stage-1 and yielded a better CSS model for further in-domain stage-3 fine-tuning and downstream ASR; (iii) real unsupervised data, regardless of the array geometry and language, was beneficial not only for the in-domain inference but also for out-of-domain ones.
