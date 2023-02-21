# Introduction

Since the STFT phase components of individual signals are not additive for multiple simultaneously active speakers, the extension of the idea of training the CNN with synthesized noise signals is not straightforward.

To be able to train a CNN for multi-speaker localization using synthesized noise signals, we propose to use the assumption that speakers are not simultaneously active per time-frequency. (i.e. W-disjoint orthogonality)[It is possible to blindly separate an arbitrary number of sources given just two anechoic mixtures provided the time-frequency representations of the sources do not overlap, a condition which we call W-disjoint orthogonality.]

# Speaker localization with CNNs

Formulate the problem of multi-speaker DOA estimation as a multi-class multi-label classification.

The number of classes, $I$, and the class vector is formed based on a discretized set of possible DOA values.

The input representation chosen in this work is the same as [*], where the phase component of the STFT coefficients of the signal are given in the form of a matrix of size $M \times K$, where $M$ and $K$ are the number of mics and freq sub-bands, respectively.

![image-20211009230148586](https://tva1.sinaimg.cn/large/008i3skNly1gv9gfutumkj60yo0cl75i02.jpg)

Use local filters of size 2 ×1, which leads to each filter learning from the phase correlations between the neighboring microphones for each frequency sub-band separately. This is done since we want to utilize the disjoint activity of speech signals for localization.

# Generating the training data

For speech signals, it is commonly assumed that the TF representation of two simultaneously active sources do not overlap. We utilize this assumption to generate training data from synthesized noise signals for two speaker localization per STFT time frame.

1. Generate the training signals for the single speaker case for different acoustic conditions.
2. For a specific source array setup, two multi-channel training signals, corresponding to different DOAs, are concatenated along the time axis.
3. For each freq sub-band separately, the TF bins for all mics are randomized to get a single training signal.

![image-20211009231526263](https://tva1.sinaimg.cn/large/008i3skNly1gv9gtzwaw0j60ym0hzn2802.jpg)

This procedure is repeated for all combinations of DOAs for all different acoustic conditions considered for training.

4. The phase map corresponding to each time frame for all training signals is extracted to form the complete training dataset.

Two important things to note regarding the randomization process:

First, it is essential that the randomization of the TF bins is done separately for each frequency sub-band, such that the order of the frequency sub-bands remains the same for different time frames. 

Secondly, it is essential that for each frequency sub-band, the TF bins for all the microphones are randomized together, such that phase relations between the microphones for the individual TF bins are preserved. ( are learned. )

Following the randomization procedure, at each time frame there are approximately equal number of TF bins with activity corresponding to the two DOAs. Therefore, at each frequency sub-band of the phase map input to the CNN, the phase of the STFT coefficients for all microphones correspond to a single source.

With this training input, the CNN can learn the relevant features for localizing multiple speakers at each time frame from the individual TF bins that contain the phase relations across the microphones for each source DOA separately.

# Experimental results

 The posterior probabilities for each DOA class obtained from the CNN output at each time frame are averaged over all the frames, and then normalized to 1. Then the final DOA estimates are obtained by choosing the DOAs corresponding to the classes with the two highest averaged posterior probabilities.



ULA with $M=4$ mics with inter-microphone distance of 8cm

transformed to the STFT domain using a DFT length of 512, with 50% overlap.

discretize the whole DOA range of a ULA with a 5° resolution to get $I=37$ DOA classes.

![image-20211010223234537](https://tva1.sinaimg.cn/large/008i3skNly1gval7votjoj60xt08f76a02.jpg)

Spatially uncorrelated Gaussian noise was added to the training signals with randomly chosen noise levels between 0 and 20 dB before extracting the phase maps.

![image-20211010223438622](https://tva1.sinaimg.cn/large/008i3skNly1gval9wjbw9j60ym0atdhl02.jpg)

cross-entropy as the loss function

Adam gradient-based optimizer

mini-batches of 512 time frames

each layer followed by a dropout with a rate of 0.5

![image-20211010225741491](https://tva1.sinaimg.cn/large/008i3skNly1gvalxvs8wlj30y90bjac8.jpg)

# Conclusion and further work

We presented a CNN-based method trained with synthesized noise signals for the task of multi-source DOA estimation by utilizing the assumption of disjoint activity of speech sources in the STFT domain.



\* BB DOA Estimation using CNN