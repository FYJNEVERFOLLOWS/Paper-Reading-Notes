# [FaSNet-TAC] End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation 阅读笔记

# Abstract
It is vital to guarantee the robustness of a system with respect to the locations and numbers of microphones in ad-hoc mic speech separation.

Propose transform-average-concatenate (TAC), a simple design paradigm for channel permutation and number invariant multi-channel speech separation. Based on the filter-and-sum network (FaSNet), a recently proposed end-to-end time-domain beamforming system, we show how TAC significantly improves the separation performance across various numbers of microphones in noisy reverberant separation tasks with ad-hoc arrays.

Moreover, we show that TAC also significantly improves the separation performance with fixed geometry array configuration, further proving the effectiveness of the proposed paradigm in the general problem of multi-microphone speech separation.

# Intro
Deep learning-based beamforming systems, sometimes called *neural beamformers*, have been an active research topic recently. A general pipeline in the design of many recent neural beamformers is to first perform pre-separation on each channel independently, and then apply conventional beamforming techniques such as MVDR or MWF (multi-channel Wiener filtering) based on the pre-separation outputs.

Another pipeline for neural beamformers is to directly estimate the beamforming filters in either time domain or freq domain, which allows for end-to-end estimation of beamforming filters in a fully-trainable fashion. But such systems typically assume knowledge about the number of mics, since a standard network layer can only generate a fix-sized output.

FaSNet directly estimates the time-domain beamforming filters without specifying the number or permutation of the mics. With a two-stage design, the first stage applies pre-separation on a selected reference mic by estimating its beamforming filters, and the second stage estimates the beamforming filters for all remaining mics based on pair-wise cross-channel features between the pre-separation output and each of the remaining microphones.