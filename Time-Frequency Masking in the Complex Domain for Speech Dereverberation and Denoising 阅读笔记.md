# [复数域语音增强] Time-Frequency Masking in the Complex Domain for Speech Dereverberation and Denoising 阅读笔记

# Abstract

This paper addresses monaural speech separation in reverberant and noisy environments. 

Enhance the magnitude and phase by performing separation with an estimate of the complex ideal ratio mask.



# Introduction

Speaker can hear not only the sound that directly reaches their ears, but also reflections off the walls, ceiling and furniture. 

These reflections, termed reverberation, are altered versions of the original speech. 

Reverberant speech consists of 3 components: the direct sound (anechoic part corresponding to the first wavefront), early reflections (arrive up to 50ms after the direct sound) and late reflections (come anytime thereafter). 

Reverberation combined with additive noise can be detrimental to the speech intelligibility of normal hearing listeners. 

A solution for removing reverberation and noise would be beneficial for a variety of speech processing tasks.

Weighted Prediction Error is an unsupervised approach that operates in the complex T-F domain and uses linear prediction to shorten the RIR, which in effect removes late reverberation. But WPE does not address noise problem.

Performing T-F masking in the complex domain (cIRM) is very beneficial when dealing with background noise.

We propose to use DNNs to learn a mapping from reverberant (and noisy) speech to the cIRM.



# Notations and definitions

$$
y(t)=h_d(t)*s(t)+h_e(t)*s(t)+h_l(t)*s(t)\\
=d(t)+y_e(t)+y_l(t)
$$

