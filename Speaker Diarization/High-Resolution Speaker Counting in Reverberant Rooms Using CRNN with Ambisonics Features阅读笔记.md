#! https://zhuanlan.zhihu.com/p/456366383
# Abstract & Conclusion

Speaker counting is the task of estimating the number of people that are simultaneously speaking in an audio recording.

Proposed a multichannel CRNN which produces an estimation at a short-term frame resolution.

The proposed multichannel CRNN yields competitive speaker counting performance at a framewise precision which can be very useful for online speech analysis task such as speaker localization or diarization.

# Introduction

Speaker counting can be seen as a subtask of speaker diarization, which consists in estimating who speaks and when, and which has long been limited to the case where one person speaks at a time, since it becomes much more complicated when several speech signals overlap.

Contributions:

1. Evaluate the benefit of using a multichannel input in a NN for the speaker counting problem. Use the Ambisonics multichannel audio format.
2. Tackle the challenging problem of estimating the number of speakers at a short-term frame resolution. This would enable a low-latency (possibly real-time) overall process such as speaker separation and localization.

# Proposed Method

**A. Input features**

Use the Ambisonics signal representation as a multichannel input.

The Ambisonics format is particularly well-suited to represent the spatial properties of a soundfield and irrelative to the mic array configuratoin.

The Ambisonics format is produced by projecting the recorded multichannel audio onto a basis of spherical harmonic functions.

The use of first-order Ambisonics (FOA) has been shown to provide a neural network with sufficient spatial information for single- and multi-speaker localization.

FOA provides a decomposition of the signal into the first four Ambisonics channels denoted $W, X,Y,Z$. Channel $W$ (order-0 spherical harmonic) represents the soundfield as if it was recorded by an omnidirectional microphone at the observation point. Channels $X,Y$ and $Z$ (order-1 spherical harmonics) correspond to the recordings of three polarized orthogonal bidirectional microphones.
$$
\left[\begin{array}{c}
W(t, f) \\
X(t, f) \\
Y(t, f) \\
Z(t, f)
\end{array}\right]=\left[\begin{array}{c}
1 \\
\sqrt{3} \cos \theta \cos \phi \\
\sqrt{3} \sin \theta \cos \phi \\
\sqrt{3} \sin \phi
\end{array}\right] p(t, f)
$$


在 $p(t,f)$ 处的 FOA 表示。

Sound pressure: $p$, the plane wave's azimuth $\theta$ and elevation $\phi$. 



16kHz, 1024-point STFT with a sinusoidal analysis window and 50% overlap.

**B. Outputs**

Consider speaker counting as a classification problem where each class corresponds to a different number of active speakers from 0 (i.e. only background noise) to a maximum of 5 active speakers. 

One-hot encoding: vector of size 6.

Softmax function is used to represent the probability distribution over the 6 classes.

The predicted number of speakers is the class with the highest output probability.

Use the categorical cross-entropy loss.

**C. Sequence-to-sequence mapping**

Target a much finer temporal resolution: predict the total number of speakers for each short-term frame.

For training, each short-term frame is labeled by the total number of active speakers within the frame.

We still want to exploit a larger local context. For each frame to classify, we use a short signal sequence of $N_t$ frames as corresponding input to the network. $N_t$ is within 10 to 30, i.e. a 320-ms to 1-s local context in our experiments.

Treat the problem as a seq-to-seq scheme, combined with one-frame shift. Each input seq of $N_t$ frames produces a synchronized seq of $N_t$ decoded class probability vectors.

**D. Network architecture**

![image-20220111163405904](https://tva1.sinaimg.cn/large/008i3skNly1gy9thfioavj30u01cwdk7.jpg)

After each conv layer, ReLU activations are used, whereas in the LSTM cells, we use tanh activation except for the recurrent step which uses hard-sigmoid.

Contrasts with [CountNet: Estimating the number of concurrent speakers using supervised learning] where seq-to-one decoding was used in the LSTM to find the max number of simultaneous speakers within a 5s audio mixture.

# Experiments

**A. Data**

10000 training rooms, 100 val / test rooms

RT60: 0.2-0.8s

TIMIT

Speech mixtures of 15s length

For each single-speaker reverbrant signal, the 15s audio is generated as follows:

![image-20220111185758194](https://tva1.sinaimg.cn/large/008i3skNly1gy9xn4gdyij318u0ggn2j.jpg)

To generate a more class-balanced dataset, the probabilities to generate a mixture (for a given room) where $N_{sp}=1,2,3,4,5$ are respectively set to 0.2, 0.3, 0.4, 0.5 and 1.

The signal-to-interference ratio (SIR) between the first source and the other sources is set randomly within 0–10 dB.

Use a random signal-to-noise ratio (SNR) between 10 and 20 dB with respect to the first source.

Using different speakers and noise sequences in the train, validation and test sets.

25 hours for training, and 0.42 hours for validation and test.



**B. Configurations**

In order to assess the influence of the temporal context on the speaker counting performance, we conducted experiments with different values for the number of frames in the input feature $X\in\mathbb{R}^{N_t\times513\times4}$. We tested $N_t=10\ (320ms),\ 20\ (640ms)\ and\ 30\ (\approx1s)$.



**C. Training procedure**

ADAM optimizer, lr 10-3, $\beta_1$=0.9, $\beta_2=0.999$, $\epsilon=$10-7

Early stopping was applied with a patience of 50 epochs by monitoring the accuracy on the validation set, and the training never exceed 300 epochs.



**D. Metrics and baseline**

Per-class classification accuracy and the MAE per class $k$:
$$
M A E(k)=\frac{1}{T(k)} \sum_{t=1}^{T(k)}|\hat{k}(t)-k|
$$
where $\hat{k}(t)$ is the predicted class for frame $t$ of ground-truth class $k$, $T(k)$ is the total number of frames of class $k$.

To assess the advantage of using multichannel features, we trained and tested the same CRNN with single-channel input features, using only the $W$ channel.



**E. Results**

Trained each neural network 10 times, then evaluate each of them on the test set and average the obtained results.

![image-20220112120501167](https://tva1.sinaimg.cn/large/008i3skNly1gyarbrcm02j328k0lkqbe.jpg)

performance gradually decreases with the increasing number of concurrent speakers.

Temporal context seems to have a slight impact on the performance.