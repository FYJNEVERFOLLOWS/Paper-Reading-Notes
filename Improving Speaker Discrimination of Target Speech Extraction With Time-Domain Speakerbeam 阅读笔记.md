# Improving Speaker Discrimination of Target Speech Extraction With Time-Domain Speakerbeam 阅读笔记

# Abstract
Previously proposed SpeakerBeam exploits an adaptation utt of the target speaker to extract his voice characteristics that are then used to guide a neural network towards extracting speech of that speaker. SpeakerBeam presents a practical alternative to speech separation as it enables tracking speech of a target speaker across utts, and achieves promising speech extraction performance. However, it sometimes fails when speakers have similar voice characteristics, such as in same-gender mixtures.

In this paper, we investigate strategies for improving the speaker discrimination capability of SpeakerBeam. First, propose a time-domain implementation of SpeakerBeam similar to TasNet. Besides, we investigate (1) the use of spatial features to better discriminate speakers when mic array recordings are available, (2) adding an auxiliary speaker identification loss for helping to learn more discriminative voice characteristics. We show experimentally that these strategies greatly improve speech extraction performance, especially for same-gender mixtures, and outperform TasNet in terms of target speech extraction.

# 1. Intro
Despite the great success of neural network-based *speech separation*, it requires knowing or estimating the number of sources in the mixture and still suffers from a global permutation ambiguity issue, i.e., an arbitrary mapping between source speakers and outputs. These limitations arguably limit the practical usage of speech separation. In contrast, *target speech extraction* exploits an auxiliary clue to identify the target speaker in the mixture and extracts only speech of that speaker. It naturally avoids the global permutation ambiguity issue and does not require knowing the number of sources in the mixtures.

FD-SpeakerBeam (Freq domain) cannot discriminate same-gender mixtures.

In this paper, 

First, propose TD-SpeakerBeam and use a convolutional network to obtain richer speaker embedding vectors.

Moreover, to further improve speaker discrimination capability, we extend TD-SpeakerBeam to accept spatial info from mic array recordings as additional input features and proposed an alternative approach, called *internal combination*, for exploiting spatial info more effectively within the SpeakBeam framework.

Finally, to enforce learning more discriminative speaker embedding vectors, we propose using a multi-task loss for training SpeakerBeam, which combines a speech reconstruction loss with a *speaker identification loss* (SI-loss).

# 2. Proposed TD-SpeakerBeam
SpeakerBeam is composed of two networks, an *extraction network*, and an *auxiliary network*. 
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20221012162654.png)
Insert an adaptation layer between the first and second convolution blocks to drive the network towards extracting the target speech. The adaptation layer accepts a speaker embedding vector of the target speaker, $\mathbf{e}^s$, as auxiliary info. 

The target speaker embedding vector, $\mathbf{e}^s$, is computed by the time-domain auxiliary network. The auxiliary network consists of an encoder layer and a single convolution block similar to those used in the extraction network.

## 2.2. Spatial features
