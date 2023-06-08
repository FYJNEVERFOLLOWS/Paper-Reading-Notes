#! https://zhuanlan.zhihu.com/p/632927392
# DNS Challenge Paper Summary
# DPCRN
SkipConvNet: Replace residual connection with SkipConv

# FRCRN
Boosting Feature Representation using Frequency Recurrence for Monaural Speech Enhancement: Sequential Memory Network on frequency dimension after each Conv block on TF map in the encoder.

# Monaural Speech Enhancement with Complex Convolutional Block Attention Module and Joint Time Frequency Losses
Boost the representation power of the convolutional layers using attention mechanism to fuse spatial and channel-wise information. Present a lightweight and general complex-valued channel-spatial attention mechanism. Proposed complex convolutional block attention module (CCBAM) which can be easily integrated into any complex-valued convolutional layers. Add CCBAM to both decoder layers and skip connections. CCBAM helps the skip connections to pass meaningful information that are beneficial for CRM estimation and helps the decoder to concentrate on 'what' and 'where' of feature maps when performing target reconstruction.

# DCCRN+
Channel-wise Subband DCCRN with SNR Estimation for Speech Enhancement
Subband processing via learnable neural filters for band split and merge, leading to compact model size and speed-up inference
Skip connection -> convolution pathway

# S-DCCRN: Super Wide Band DCCRN with learnable complex feature for speech enhancement
wide-band: 16k Hz
super wide band: 32k Hz
full-band: 48k Hz
Compressed feature like bark spectrum for Hi-Fi speech enhancement

Propose two lightweight DCCRN sub-modules for sub-band and full-band (SAF) modeling separately, since it is considered that low frequency bands contain higher energy while higher frequency bands have a great impact on subjective perception.
skipconnect -> convolution pathway (DCCRN+)
Propose Learnable Spectrum Compression.
Employ a complex feature encoder (CFE) after STFT and a complex feature decoder (CFD) before iSTFT motivated by DPT-FSNet.
The compression rate of frequency bands should be different since high-frequency bands may require a lower compression ratio to maintain its high energy.

# Spatial-DCCRN
DCCRN Equipped with Frame-level Angle Feature and Hybrid Filtering for Multi-channel Speech Enhancement
sub-channel full-channel processing; angle feature extraction (conv2d and Dense block on cosIPDs); masking and mapping filtering