# [时域多通道] Inter-channel Conv-TasNet for multichannel speech enhancement 阅读笔记

# Abstract
Studies on the efficient multichannel network structure fully exploiting spatial information and inter-channel relationships is still in its early stages.

Propose an end-to-end time-domain speech enhancement network that can facilitate the use of inter-channel relationships at individual layers of a DNN.

# Introduction
2 main characteristics of IC Conv-TasNet:
1. Constitute a 3-D tensor by stacking multichannel encoder outputs in the channel dimension. 
2. Separate the roles of the depthwise and 1-D convolutions of the TCN such that the depthwise convolution extracts inter-channel relationships only, whereas the 1-D Conv layer focuses on the extraction of spectral and temporal features. To this end, the 1-D Conv layer is replaced by a 2-D Conv functioning in the feature and time dimensions.
   
# Model Architecture
The input waveform is divided into $L$ overlapping segments of window length $K$. The 1-D Conv layer in the encoder module converts each segment data into a feature vector of length $F$, i.e. [L, K] -> [L, F].

The encoder output is fed into the mask estimation network (separation module) to estimate the source separation mask.

The separation module consists of a single $1\times1$ Conv layer (bottleneck layer) compressing the number of features from $F$ to $N$ ($F > N$), followed by several stacks of 1-D Conv blocks.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220810112033.png)

**Each 1-D Conv block implements its own expansion and compression of features before and after the D-Conv (dilated depthwise conv)**

The serial connection of 1-D Conv blocks constituting a single TCN stack increases the receptive field by using a higher dilation factor $d$. Multiple stacks are used to extract different information from multiple skip connections.

MC Conv-TasNet incorporated one modification into SC Conv-TasNet: the encoder is extended in the channel dimension and 1-D Conv operations are applied to individual microphone channels. See Fig 3(a). Thus is compatible with other layers of SC Conv-TasNet. However, this superposition of multichannel output causes the inter-channel relationship to be lost in the following conv layers.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220810112947.png)

## 2-D Conv-TasNet
To avoid the loss of spatial information accompanied by the addition operation of the MC Conv-TasNet, we concatenate the multichannel encoder outputs, which yields a 2-D tensor $\bold{w}\in\mathbb{R}^{L\times FM}$. Additionally, the bottleneck layer is modified to downsize the number of features (FM) to the number of features (N). Consequently, the channel and feature info are mixed by the $1\times1$ Conv layer before TCN.

Modified encoder and bottleneck layer compared to SC Conv-TasNet.

One extra modification: select one microphone channel feature $\bold{w}_{ref}=\mathbb{R}^{L\times F}$ among $\bold{w}$ as mixure and multiply the generated mask to that channel only.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220810112910.png)

## 3-D Conv-TasNet
A 3-D tensor of size (L,F,M) is constructed by stacking the outputs of the multichannel encoder and changing its size to (L,N,C) through two 1x1 Conv layers positioned before TCN. By constructing a 3-D tensor and applying separate 1x1 Conv operations along the channel and feature dimensions, we attempt to independently treat the channel and feature info throughout the entire TCN layer.

## IC Conv-TasNet
Unlike the 3-D Conv-TasNet that utilizes the inter-channel relationship at the mask generation stage, the IC Conv-TasNet extracts inter-channel features within the TCN layers to fully exploit the available spatial info.

Compared to 3-D Conv-TasNet, IC Conv-TasNet modified the TCN blocks of the mask estimation network.

Fig. 2(c), the size of the channel dimension is increased by 1x1 Conv from $C$ to the number of hidden layers $H=4C$. [This modification promotes channel-wise diversity without changing feature and time dimensions.]

The D-Conv layer is modified to apply 2-D depthwise convolutions in the feature and time dimensions

Unlike other Conv-TasNets that apply 1x1 conv layers before the skip and residual paths in the feature dimension, the 1x1 conv layers of IC-Conv-TasNet only extract the info in the channel dimension to focus on inter-channel relationships.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220810131938.png)

After the skip connection is connected to the PReLU, the channel info is compressed through 1x1 conv (C) to make the feature map into a single-channel. 1x1 conv (N) adjusts the number of features from $N$ to that of the encoder output $F$.

# Experiment
## Experiment procedure
Used the MC Conv-TasNet as the baseline model for the performance comparison. The first exp was to compare the performance of the three variants of Conv-TasNet. The second exp was to search for the best model parameters through various parameter studies (changing the number of layers in each stack of TCN (D), number of TCN stacks (S), number of features (F, N), and the size of channel dimension (C)).
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220810150808.png)

SDR loss (enhancement task) for 200 epochs. 

Encoder and decoder processed the temporal waveform using a window with a length of 256 and overlap of 50%. Used 3x3 kernel size for the 1-D or 2-D Conv block of TCN. Channel 5 was selected as the reference encoder output. ADAM with a lr of $10^{-3}$.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220810153528.png)
# Results
## Changing the model architecture
Table 2 illuminates that the 2-D C-T outperforms the baseline MC C-T.
In the MC Conv-TasNet, the spatial information in the multichannel output from the encoder is lost by summing all channel outputs into a single-channel, whereas the 2-D Conv-TasNet preserves it by concatenating all encoder outputs.

Main diff between the 2-D and 3-D Conv-TasNet is the separation of the channel and feature dimensions. Because the channel-feature size product ($C\times N$) of 3-D C-T was equal to the total feature size ($N$) of 2-D C-T, the performance enhancement is solely attributable to the separation of the channel dimension.

3-D C-T maintain channel-dependent information through convolution operations.

Nevertheless, the 3-D C-T is limited in that different channel data are not mixed by the 1-D Conv layer. The TCN structure of the 3-D C-T is merely a parallel connection of multiple TCNs for individual channel data. The inter-channel relationships are only utilized at the final mask generation step by the 1x1 conv(C).

This limitation of 3-D C-T can be resolved by IC C-T. IC C-T has an improved TCN structure that aggregates the $C$ channel data into $H$ hidden layers, and subsequently the inter-channel relationships between them are extracted by the 1x1 Conv layers.

## Parameter study
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220810155907.png)

Even with a 25% feature size (F), performance degradation is not considerable.

From Model 6 to Model 7, noticeable improvements due to the increment of N (the number of features for the TCN input). The TCN features rather than the encoder outputs strongly influenced the performance.

From Model 7 to Model 8, F is not so useful.

Best configuration: a small-sized encoder output (F=512) with a high number of features for the TCN input (N=128).

From Model 7 to Model 9, the number of channels is vital.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202208/20220810162920.png)

# Conclusion
The proposed IC C-T introduced 3 major modifications to the conventional C-T to effectively learn spatial info. First, in the encoder part, encoded multichannel signals were stacked along the new channel dimension to form a 3-D tensor, rather than summing to a single output channel. Therefore, the spatial info could be analyzed using the channel dimension, independent of the feature dimension. Second, the 1-D dilated convolution with respect to the feature dimension in TCN was changed to a 2-D dilated convolution for the feature and time dimensions. Thus, the channel-dependent info could be separately processed, which was implemented by 1x1 conv layers that operated in the channel dimension rather than the feature dimension of the original C-T.