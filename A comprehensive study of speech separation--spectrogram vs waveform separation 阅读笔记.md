#! https://zhuanlan.zhihu.com/p/583840999
# A comprehensive study of speech separation: spectrogram vs waveform separation 阅读笔记

# Abstract
We incorporate effective components of the TasNet into a freq-domain separation method. We introduce a solution for directly optimizing the separation criterion in freq-domain networks. 

Our exp results show that spectrogram separation can achieve competitive performance with better network design.

# 1. Intro
Investigate two approaches of speech separation; i.e., spectrogram (freq-domain) and waveform (time-domain) separation in detail.

Incorporate effective components of TasNet into a spectrogram separation framework, such as CNN-based separation network as well as loss function. 

# 2. Problem Setup
## 2.1. Speech Separation: Spectrogram vs Waveform
Our focus in this paper is on masking based separation.

[End-to-end speech separation with unfolded iterative phase reconstruction] proposed iterative phase reconstruction algorithm which has shown to perform better than using the mixture phase for reconstruction of the signal.

Instead of mapping samples into the freq-domain, TasNet maps them into a new temporal resolution space. The encoder does not analyze time samples into magnitude and phase components, it operates on a unified representation which makes it different from the spectrogram separators.

## 2.3. Metric
$$
\begin{gathered}
\mathrm{SDR}=10 \log _{10} \frac{\left\|x_{\text {target }}\right\|^2}{\left\|e_{\text {inter }}+e_{\text {noise }}+e_{\text {artif }}\right\|^2}, \\
\text { Si-SNR }=10 \log _{10} \frac{\left\|x_{\text {target }}\right\|^2}{\left\|e_{\text {noise }}\right\|^2},
\end{gathered}
$$

$x_{target}=\frac{\langle x, \hat{x}\rangle x}{\|x\|^2}$

For Si-SNR, the scale invariant is guaranteed by mean normalization of estimated and reference signals to zero mean.

# 3. Improved Spectrogram Separation
Traditional TF-masking/spectrogram separation networks are different from TasNet in 3 main components: 1) *Encoder-decoder*: STFT/iSTFT vs. Conv-1d/ConvTranspose-1d. The latter one does not decompose the signal into mag and phase components (part of the gain achieved from waveform separation can be related to this fact). 2) *Separation network*: CNN-based structure outperforms the BLSTM-based network for their task. 3) *Training loss*: MSE for TF masking which does not optimize the separation criterion directly. TasNet is trained with Si-SNR which directly optimizes the separation performance.

In this paper, we incorporate the CNN-based separation structure of TasNet into a spectrogram separation framework to examine the effectiveness of the network structure.

## 3.2. Si-SNR
To the best of our knowledge Si-SNR loss was not used with freq domain frameworks before.

STFT can be formulated simply as a Conv-1d operation with a fixed specific kernel function, and iSTFT as well can be simply implemented with ConvTranspose-1d. We use the fixed kernel with hamming window.

# 4. Multi-channel Speech Separation
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221114151752.png)

$$
A_{s, t, f}=\sum_{i=1}^6 \frac{e_s^{i, f} \frac{Y_{i_1, t, f}}{Y_{i_2, t, f}}}{\left|e_s^{i, f} \frac{Y_{i_1, t, f}}{Y_{i_2, t, f}}\right|},
$$

$e_s^{i,f}$ represents steering vector coefficient of speaker $s$ DOA at mic $i$ for freq $f$. In our experiments, we append both speaker angle features together, and always assume the first one is the target speaker for optimizing the target extraction loss.

# 5. Exp
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221114152846.png)
Results in Table. 2 confirm the effectiveness of the Si-SNR loss function over MSE.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221114152920.png)
