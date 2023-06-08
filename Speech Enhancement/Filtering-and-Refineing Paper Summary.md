#! https://zhuanlan.zhihu.com/p/635183845
# Filtering-and-Refining Paper Summary
Masking and Inpainting: A Two-Stage Speech Enhancement Approach for Low SNR and Non-Stationary Noise

Glance and gaze: A collaborative learning framework for single-channel speech enhancement
PHASEN estimates the sine and cosine of the phase

Filtering and Refining: A Collaborative-Style Framework for Single-Channel Speech Enhancement
The residual term between target and coarse spectrum (enhanced magnitude coupled with noisy phase) is utilized to repair the phase, because this term itself has a harmonic-like structure and its dynamic range is effectively shrunken compared with clean RI parts. Truncate the output mag mask range into (0, 1) with the sigmoid function to facilitate training convergence, and the remaining regions with mask values exceeding 1 can be handled by the other branch.