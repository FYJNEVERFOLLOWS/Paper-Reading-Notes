# Universal Speaker Extraction in the Presence and Absence of Target Speakers for Speech of One and Two Talkers 阅读笔记

# Abstract
Traditional speaker extraction models fail in scenarios when the target speaker is absent from the mixture.

Propose to handle speech mixtures with one or two talkers in which the target speaker can either be present or absent.

SE uses a spk's ref signal to extract the target spk's voice in a multi-talker speech mixture w/o any prior knowledge about the number of speakers.

In the presence of the target speaker, the model extracts the target speaker’s voice, and in the absence of the target speaker, the model is expected to output silence. We intro a joint training scheme with one unified loss func for all four conditions.

# Universal Speaker Extraction Conditions
A universal speaker extraction system should perform under four conditions to cover all acoustic scenarios in everyday conversational situations.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202303/1677740376897.png)

Speaker extraction models have been mostly trained on two- or multi-talker scenarios in the presence of the target speakers, denoted as 2T-PT. Those models usually fail when being applied to one of the other three scenarios, where they show poor performance, extract artifacts, or recover a random speaker's voice. The failure is not desired. Rather the model shall either extract the target speaker's voice (PT) or silence (AT).

# Joint Training Scheme
Propose a joint training scheme with a unified loss function as we are building one system for all four conditions.

## Metric stabilization
Commonly a little stabilization value $\varepsilon = 1e^{-8}$ is introduced to prevent the equation from breaking.
$$
\text{SI-SDR} \approx 20 \log _{10}\left(\frac{\left\|\frac{\hat{s}^T s}{\|s\|^2+\varepsilon} s\right\|}{\left\|\frac{\hat{s}^T s}{\|s\|^2+\varepsilon} s-\hat{s}\right\|+\varepsilon}+\varepsilon\right)
$$
To evaluate the quality of the reconstructed silence, define silence-evaluating SI-SDR (SE-SI-SDR) as:
$$
\text{SE-SI-SDR} = 20 \log _{10}\left(\frac{\left\|\frac{\hat{s}^T s}{\|s\|^2+\varepsilon} s\right\|+\varepsilon}{\left\|\frac{\hat{s}^T s}{\|s\|^2+\varepsilon} s-\hat{s}\right\|+\varepsilon}\right)
$$
This still preserves the stability for $s=0$ but reflects the reconstructed signal $\hat{s}$ in the calculation.

# Exp setup
Conduct exp on diff training schemes and database compositions, while maintaining the same network architecture for fair comparisons.

