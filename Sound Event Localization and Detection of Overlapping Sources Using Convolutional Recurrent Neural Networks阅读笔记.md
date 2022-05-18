#! https://zhuanlan.zhihu.com/p/447495306
# Sound Event Localization and Detection of Overlapping Sources Using CRNN阅读笔记
# Abstract & Conclusion

Proposed a CRNN for joint sound event localization and detection, taking a seq of consecutive spectrogram time-frames as input and maps it to two outputs in parallel. 

The usage of such non-method-specific feature (phase and magnitude spectrogram) makes the method generic and easily extendable to different array structures.

# Introduction

SED aims at detecting temporally the onsets and offsets of sound events and further associating textual labels to the detected events.

The SED task in literature has most often been approached using different supervised classification methods that predict the framewise activity of each sound event class.



Most previous methods relied on a single array or distributed arrays of omnidirectional microphones, captured source location information mostly in phase- or time-delay differences between the microphones.

Compact microphone arrays with full azimuth and elevation coverage, such as spherical microphone arrays, rely strongly on the directionality of the sensors
to capture spatial information, this reflects mainly in the magnitude differences between channels. Motivated by this fact we proposed to use both the magnitude and phase component of the spectrogram as input features in [Direction of arrival estimation for multiple sound sources using convolutional recurrent neural network].

inter-aural level difference (ILD), inter-aural time difference (ITD), GCC or eigenvectors of spatial covariance matrix are not generic to array configuration.



**D. Contributions of this paper**

Two broad areas: the proposed SELD method, and the exhaustive evaluation studies presented.

1. The proposed SELD is the first method that addresses the problem of localizing and recognizing more than two overlapping sound events simultaneously and tracking their activity with respect to time. Robust to unseen spatial locatoins, reverb and ambient. Generic enough to learn to perform SELD from any input array structure.
2. Present the performance of the proposed method with respect to various design choices such as the DNN architecture, input feature and DOA output format. Also present the comprehensive results of the proposed method with respect to six baselines evaluated on seven datasets with different acoustic conditions, array configurations and the number of overlapping sound events.

# Method

At the first output, SED is performed as a multi-label classification task, allowing the network to simultaneously estimate the presence of multiple sound events for each frame. At the second output, DOA estimates in the continuous 3D space are obtained as a multi-output regression task, where each sound event class is associated with three regressors that estimate the 3D Cartesian coordinates $x$, $y$ and $z$ of the DOA on a unit sphere around the microphone.

![image-20211117151007583](https://tva1.sinaimg.cn/large/008i3skNly1gwi5z2oe3yj30md0ky765.jpg)

![image-20211117151107889](https://tva1.sinaimg.cn/large/008i3skNly1gwi6041echj30ly0mn76h.jpg)

**A. Feature extraction**

$M$-point DFT on Hamming window of lenth $M$ and 50% overlap

Only the $M/2$ positive frequencies without the zeroth bin are used

C channels



**B. Neural network architecture**

Each CNN layer has $P$ filters of $3\times3\times2C$ dimensional receptive fields.

The dimensionality is reduced using max-pooling ($MP_i$) along the frequency axis, thereby keeping the sequence length T unchanged.

Reshape CNN output $T\times2\times P$ to $T \times2P$ before fed to bidirectional RNN layers.

 Each layer has $Q$ nodes of GRU with tanh activations.

The DOA branch has $3N$ nodes where each of the $N$ sound event classes is represented by 3 nodes (x, y, z).

x, y, z's range: [-1, 1], thus used the tanh activation

The SED output of the SELDnet is in the continuous range of [0, 1] for each class, while the DOA output is in the continuous range of [−1, 1] for each axes of the sound class location.

The network hyperparameters are optimized based on cross-validation.



**C. Training procedure**

binary cross-entropy loss for the SED predictions

MSE loss for the DOA estimates
$$
MSE = \frac{(x_1 - x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2}{3}
$$
**The angles are discontinuous, while the Cartesian coordinates are continuous.**

**This continuity allows the network to learn better.**

A weighted combination of MSE and binary cross-entropy loss for 1000 epochs

Adam optimizer, Early stopping

Training is stopped if the SELD score on the test split does not improve for 100 epochs.

# Evaluation

**B. Baselines**

DOAnet & Multi-speaker localization using CNN trained with noise (only learn phase difference)



**C. Evaluation metrics**

![image-20211117190031973](https://tva1.sinaimg.cn/large/008i3skNly1gwicmu09yvj30mh0g8q5r.jpg)

# Results and Discussion

![image-20211118090908962](https://tva1.sinaimg.cn/large/008i3skNly1gwj15um24bj319z0foq7v.jpg)

An ideal method has a frame recall of one (reported as percentages in Table) and DOA error of zero.
$$
frame recall = \frac{TP}{TP+FN}
$$
TP: total number of time frames in which the number of DOAs predicted is equal to reference.

FN: total number of time frames where the predicted and reference DOA are unequal.
