# Complex neural spatial filter: Enhancing multi-channel target speech separation in complex domain 阅读笔记

## SPL 2021
https://arxiv.org/abs/2104.12359

# Abstract
The existing models for estimating cRM are designed in the way that the real and imaginary parts of the cRM are separately modeled using real-valued training data pairs. The research motivation of this study is to design a deep model that fully exploits the temporal-spectral-spatial information of multi-channel signals for estimating cRM directly and efficiently in complex domain.

The idea is triggered by *complex DNN* models [DCCRN]: single-stream model instead of two-stream for real and imag parts is designed where all the network components and operations are in complex domain (such as Complex batch normalization).

$$
\begin{aligned}
\operatorname{cBLSTM}\left(X_r, X_i\right) & =\left(\operatorname{BLSTM}_r\left(X_r\right)-\operatorname{BLSTM}_i\left(X_i\right)\right) \\
& +j\left(\operatorname{BLSTM}_r\left(X_i\right)+\operatorname{BLSTM}_i\left(X_r\right)\right)
\end{aligned}
$$



