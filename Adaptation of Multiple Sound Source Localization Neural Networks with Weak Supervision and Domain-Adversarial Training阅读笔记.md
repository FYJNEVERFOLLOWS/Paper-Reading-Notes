Two domain adaptation methods are investigated: weak supervision and domain-adversarial training.

# Introduction

DNN-based approaches suffer two major drawbacks:

1. The costly data recording and annotation process hinders the application of DNN-based SSL systems.
2. Their sensitivity to the mismatch between the training and test conditions. The acoustic environments vary considerably in terms of bg noise, reverb, SNR as well as distribution of source locations. Larger mismatch between the virtual and real environments when using simulated training data (difference in sensor properties, device physical bodies, etc).

By using simulation, we can easily produce sufficient training data for any device. We can acquire a large amount of unlabelled or weakly labelled real data, which can be exploited for adaptation.

We propose weak supervision by output regularization. Specifically, we examine the weak supervision with known number of sources. The number of sources contains crucial information for SSL, and is much easier to annotate compared to the exact location of each source. Based on the available weak labels, we can significantly reduce the dimension of the desired output space, and the output regularization aims to bring the network output closer to the reduced space.

# Relation to prior work

In this paper, we focus on the adaptation of multiple sound source localization neural networks. Specifically, our approach differs from previous studies in: (1) the neural net- work for adaptation does not predict posterior probability; (2) in addition to unsupervised adaptation, we also investigate weakly supervised adaptation to produce more stable results; (3) we examine the regularization not only on the output, but also on the features.

# Proposed domain adaptation approach

