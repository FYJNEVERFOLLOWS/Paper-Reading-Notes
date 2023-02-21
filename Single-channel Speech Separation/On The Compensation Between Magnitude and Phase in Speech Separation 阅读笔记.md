# On The Compensation Between Magnitude and Phase in Speech Separation 阅读笔记

## SPL 2021
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202302/20230208161435.png)

# Abstract
Many recent studies optimize loss funcs defined solely in the time or complex domain, w/o including a loss on mag. They produce better scores if the evaluation metrics are objective time-domain metrics and worse scores on speech quality and intelligibility metrics and usually lead to worse ASR performance, compared with including a loss on mag.

# Intro
One observation by many studies: loss on mag produce clear improvs on ASR with slightly worse SI-SDR.

Our insight is that, since phase is always difficult to estimate accurately, if the loss is defined solely in the complex or time domain, the mag of the estimated speech will tend to compensate for an inaccurate phase estimate.

# E2E Speech Separation
## A. Complex-domain Separation
$$
\mathcal{L}_{\mathrm{RI}}=\|\hat{R}-\mathrm{Real}(S)\|_1+\|\hat{I}-\mathrm{Imag}(S)\|_1
\tag{2}
$$

A mag loss can be added:
$$
\mathcal{L}_{\mathrm{RI+Mag}}=\mathcal{L}_{\mathrm{RI}}+\||\hat{R}+j\hat{I}|+|S|\|_1
\tag{3}
$$
where $|\cdot|$ extracts mag.

Train through iSTFT and define the loss in the time domain:
$$
\mathcal{L}_{\mathrm{RI-iSTFT}}=\|\mathrm{iSTFT}(\hat{S})-s\|_1
\tag{4}
$$

To improve mag of the final signal listeded by end users $\mathrm{iSTFT}(\hat{S})$, a mag loss can be included:
$$
\mathcal{L}_{\mathrm{RI-iSTFT+Mag}}=\mathcal{L}_{\mathrm{RI-iSTFT}}+\||\mathrm{STFT}(\mathrm{iSTFT}(\hat{S}))|-|S|\|_1
\tag{5}
$$
An alternative computes the mag loss before iSTFT:
$$
\mathcal{L}_{\mathrm{Mag+RI-iSTFT}}=\||\hat{R}+j\hat{I}|+|S|\|_1+\mathcal{L}_{\mathrm{RI-iSTFT}}
\tag{6}
$$

## B. Time-domain Separation
Time-domain separation implicitly models mag and phase through E2E optimization.

$\mathcal{L}_\mathrm{Wav}=$L1, L2 or log-compressed versions or SI-SDR

Incorporate a mag loss:
$$
\mathcal{L}_{\mathrm{Wav+Mag}}=\mathcal{L}_{\mathrm{Wav}}+\||\mathrm{STFT}(\hat{s})-|S|\|_1
\tag{8}
$$

# Compensation between Mag and Phase
## A. The Compensation Problem

## B. Mag Spec Approximation
This compensation view suggests that, in cases where we only need a good estimated magnitude and do not have to estimate or leverage phase, it may be better not modelling magnitude and phase simultaneously. One such scenario is robust ASR based on monaural speech enhancement, where the recognition model typically only considers mag features.

One potential issue with MSA is that when signal re-synthesis is needed, the mixture phase is typically used together with the estimated mag.

Why ASR features from estimated mags rather than from re-synthesized signals produce better ASR results? - No compensated mag, no STFT consistency.

MSA produce good mag, PESQ, STOI and WER, but worse SI-SDR (not clean phase) [good for human listening, as the human auditory system is not sensitive to slight signal shift]

# Exp

We point out that SI-SDR is very sensitive to signal shift.
