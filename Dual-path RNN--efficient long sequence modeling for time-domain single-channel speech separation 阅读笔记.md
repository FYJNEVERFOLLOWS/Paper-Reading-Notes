#! https://zhuanlan.zhihu.com/p/534750766
# [时域语音分离] DPRNN: long sequence modeling for time-domain single-channel speech separation 阅读笔记

# Abstract
The time-domain separation systems often receive input sequences consisting of a huge number of time steps, which introduces challenges for modeling extremely long sequences.

Conventional RNNs are not effective for modeling such long sequences due to optimization difficulties, while 1-D CNNs cannot perform utterance-level sequence modeling when its receptive field is smaller than the sequence length. 

Propose DPRNN that organizes RNN layers in a deep structure to model extremely long sequences. DPRNN splits the long sequential input into smaller chunks and applies intra- and inter-chunk operations iteratively.

# Intro
Time-domain separation: *adaptive front-end* and *direct regression* approaches

*Adaptive front-end* approaches aim at replacing the STFT with a differentiable transform to build a front-end that can be learned jointly with the separation network. Being independent of the traditional time-freq analysis paradigm, these systems are able to have a much more flexible choice on the window size and the number of basis functions for the front-end.

*Direct regression* approaches learn a regression function from an input mixture to the underlying clean signals without an explicit front-end. [Wave-U-Net]

The intuition of DPRNN is to split the input sequence into shorter chunks and interleave two RNNs, an *intra-chunk* RNN and an *inter-chunk* RNN, for local and global modeling. In a DPRNN block, the intra-chunk RNN first processes the local chunks independently, and then the inter-chunk RNN aggregates the info from all the chunks to perform utterance-level processing.

The DPRNN blocks iteratively and alternately perform the intra- and inter-chunk operations. This design allows the length of each RNN input to be proportional to the square root of the original input length. [When chunk size == num of chunks, the two RNNs have a sublinear input length $O(\sqrt{L})$ as opposed to the original input length $O(L)$, which greatly decreases the optimization difficulty that arises when $L$ is extremely large.]

# DPRNN
## 2.1. Model design
DPRNN consists of three stages: segmentation, block processing, overlap-add.

![](https://tva1.sinaimg.cn/large/e6c9d24ely1h3h2cyw79nj21va0qoahv.jpg)

