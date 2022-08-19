#! https://zhuanlan.zhihu.com/p/535471374
# [时域语音分离] DPTNet: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation 阅读笔记

Propose a dual-path transformer network (DPT-Net) for end-to-end speech separation, which introduces direct context-awareness in the modeling for speech sequences. By introduces a improved transformer, elements in speech sequences can interact directly, which enables DPTNet model for speech sequences with direct context-awareness. The improved transformer in our approach learns the order information of the speech sequences without positional encodings by incorporating a RNN into the original transformer.

# Introduction
Two categories: T-F domain methods and end-to-end time-domain approaches.

The dominant SS models are based on RNN or CNN, which cannot model the speech seqs directly conditioning on context, leading to suboptimal separation performance.

For example, RNN based models need to pass info through many intermediate states. And the models based on CNN suffer from the problem of limited receptive fields. The transformer based on self-attention mechanism can resolve this problem effectively, in which elements of the inputs can interact directly. Nevertheless, the transformer usually only deals with sequences with length of hundreds, while end-to-end time-domain speech separation systems often model extremely long input seqs, which can sometimes be tens of thousands. Dual-path network is an effective method to deal with this problem [DPRNN].

Contributions:
1. The first work that introduces direct context-aware modeling into speech separation. 
2. Integrate a RNN into original transformer to make it can learn the order info of the speech seqs without positional encodings. Then we embed this improved transformer into a dual-path network, which makes our approach efficient for extremely long speech sequence modeling.

# SS with DPTNet
Consists of three stages: encoder, separation layer and decoder.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202206/20220627111003.png)

## 2.1. Encoder
1-D convolution module with $N$ filters

## 2.2. Separation layer: DPTNet
DPTNet is composed of three stages: segmentation, dual-path transformer processing and overlap-add, which is inspired by the common DPRNN.

### 2.2.1. Segmentation
chunk size: $K$

hop size: $H$

All the chunks are concatenated to be a 3-D tensor $D\in R^{N\times K\times P}$

### 2.2.2. Dual-path transformer processing
Transformer is composed of an encoder and a decoder. They share the same model structure, except that the decoder is a left-context-only version for generation.

The transformer in this paper refers specially to the encoder part, and it is comprised of three core modules: scaled dot-product attention, multi-head attention and position-wise feed-forward network.

The final output of the scaled dot-product attention module is computed as a weighted sum of the values, where the weight for each value is computed by a attention function of the query with the corresponding keys. 

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202206/20220627111111.png)
Multi-head attention is composed of multiple scaled dot-product attention modules. Firstly, it linearly maps the inputs $h$ times with different, learnable linear projections to get parallel queries, keys and values respectively. Then the scaled dot-product attention is performed on these mapped queries, keys and values simultaneously. 


![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202206/20220627111154.png)

To learn the order information, we replace the first FC layer with a RNN in the Feed Forward network.

Each DPT consists of intra-transformer and inter-transformer, which are committed to modeling local and global info respectively.

This structure makes each element in speech seqs interact directly with only some elements and interact with the rest elements through an intermediate element, which allows our approach to model for extremely long speech sequences efficiently despite a slight negative impact on the direct context-aware modeling.

# Experiment
## 3.1. Dataset
WSJ0-2mix, LS-2mix

[A Pytorch implementation](https://github.com/ujscjj/DPTNet)

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202206/20220629165748.png)

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202206/20220629165814.png)

# Conclusion
DPTNet models the speech sequences directly conditioning on context and can learn the order info in speech seqs without positional encodings and model effectively for extremely long sequences of speech signals.