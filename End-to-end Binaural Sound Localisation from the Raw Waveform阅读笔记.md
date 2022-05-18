#! https://zhuanlan.zhihu.com/p/447507977
# End-to-end Binaural Sound Localisation from the Raw Waveform阅读笔记
# Abstract & Conclusion

Instead of employing hand-crafted features commonly employed for binaural sound localisation, such as the interaural time and level difference, our end-to-end system approach uses a CNN to extract specific features from the waveform that are suitable for localisation.



The WaveLoc-GTF is inspired by the auditory system and employs a conv layer that is largely based on a gammatone filterbank.

The WaveLoc-CONV employs a data-driven approach, where a conv layer with trainable 1-D kernels is dedicated for freq analysis.

# Introduction

Instead of an explicit feature extraction stage, the proposed *WaveLoc* uses a CNN with a cascade of convolutional layers to implicitly extract features directly from the raw waveform for sound localisation.

One of the key stages in the network is the frequency analysis, and two different approaches are investigated. 

1. The first approach is auditory-inspired and uses a convolutional layer based on the gammatone filterbank. The gammatone filter is a widely-used model of auditory frequency analysis, with bandwidths set to reproduce human critical bandwidths.

2. In the second model, we adopt a standard convolutional layer which is intended to learn how to perform frequency analysis along with the training process of the entire network. After frequency analysis, further convolutional layers with 2-D kernels operates directly on the signals from both ears to extract features that are similar to the binaural cues used by the auditory system. The extracted features are finally concatenated and used as input to a DNN with fully connected layers, in order to map them to the corresponding source azimuth.

# SYSTEM DESCRIPTION

![image-20211217164731020](https://tva1.sinaimg.cn/large/008i3skNly1gxgxdos38uj31u20u00yk.jpg)

The ear signals are sampled at 16kHz and framed with 20ms window size with 10ms overlap. In each frame the left and right channels are stacked together to form an input matrix of size $2\times320$.

Classification task, that's why Softmax is used.

**2.2. WaveLoc-GTF**

**Conv2D architecture: C: 32, H: 2, W: 320**

The WaveLoc-GTF system can be broadly divided into three stages: (i) a frequency analysis stage that takes the framed binaural ear signals as input, (ii) a feature extraction stage with a cascade of convolutional layers to extract suitable features for sound localisation, and (iii) a sound localisation stage based on several dense layers to perform sound localisation as a classification task.

Gammatone filter bank consists of 32 filters spanning between 70 and 7000Hz with peak gain set to 0dB. These filters are directly coded into *non-trainable* CNN kernels of size $1\times320$ with a linear activation function.

The gammatone impulse response is given by:
$$
w[t]=a t^{n-1} \cos (2 \pi f t+\phi) e^{-2 \pi b t}
$$
t: time, a: amplitude, f: the centre frequency, $\phi$: the phase of the carrier, n: the filter's order, b: the filter's bandwidth

The kernel / conv operation is defined as:
$$
y[t]=\sum_{m=-M}^{M} x[m] w[t-m]
$$
x: the input signal, w: the weights of the filter, t: the index of the actual value, M: the filter length

In each freq band, the resulting feature maps share the same dimensions ($2\times320$) of the input matrix.

Normalisation layer looks for the maximum absolute value across all the gammatone channels before dividing them by this value. Hence, the output feature values range between [-1, 1], which are further processed with $1\times2$ max pooling.

Followed by a conv block with 2-D kernels of size $2\times18$, $1\times4$ max pooling, ReLu and a conv block with 1-D kernels of size $1\times6$, $1\times4$ max pooling, ReLu.

Finally, the processed channels are concatenated and fed into two FC dense layers which consists of 1024 hidden units with ReLU activation and a dropout rate of 0.5.

The output layer consists of 37 nodes corresponding to the 37 azimuth classes, with softmax activation.

**2.3. WaveLoc-CONV**

**Conv2D architecture: C: 64, H: 2, W: 320**

WaveLoc-CONV employs a single conv layer dedicated to frequency analysis. Key difference from WaveLoc-GTF is that the freq analysis of this model is learnt during the training process together with other params of the network. 

Replace the gammatone filter bank with a conv layer with 64 1-D kernels of shape $1\times256$ as time domain filters for frequency analysis. The conv layer is followed by $1\times2$ max pooling with a linear activation function.

Instead of acting separately for each channel as in  WaveLoc-GTF, the WaveLoc-CONV processes all the output of the freq analysis stage in the next two conv layers.

Followed by a conv layer with 64 2-D kernels of size $2\times18$ to look for correlations between the left and right channels and a conv layer with 64 1-D kernels of size $1\times6$. Both layers use the ReLU and followed by $1\times4$ max pooling.

Finally, the outputs are flattened and fed into a two FC hidden layers with 1024 units each. The output layer uses softmax activation with 37 neurons.

# EVALUATION

**3.1. Binaural simulation**

The training dataset was obtained by randomly selecting 24 sentences per azimuth from the TIMIT training subset.

Another 6 sentences composed the validation dataset.

15 more sentences per azimuth were selected from the TIMIT test subset to create the test dataset.

**3.2. Experimental setup**

Adam optimiser, lr: 10e-3, batch size: 128, 50 epochs, early stopping was applied if no improvement was observed on the validation set for more than 5 epochs.

A decreasing lr was employed to improve training, being multiplied by 0.2 if no lower error was achieved after 2 epochs.

The evaluation results are reported based on chunks. Each chunk is 250ms long (25 frames).

The accuracy of the models was finally measured in terms of RMSE given in degrees.

# RESULTS AND DISCUSSION

**4.1. Anechoic training**

![image-20211218151643778](https://tva1.sinaimg.cn/large/008i3skNly1gxi0djg2rrj318g0dcgo9.jpg)

train with anechoic signals only, test on all the reverberant rooms (Room A, Room B, Room C, Room D)

![image-20211218152044091](https://tva1.sinaimg.cn/large/008i3skNly1gxi0hnn86bj31680a4jt6.jpg)

**4.2. Multiconditional training**

MCT can mitigate overfitting and increase the robustness of sound localisation in reverberant conditions. This can be done by adding either diffuse noise or reverberation to the training signals.

For each one of the four reverberant room under evaluation, all the remaining three were included for MCT.

![image-20211218152449230](https://tva1.sinaimg.cn/large/008i3skNly1gxi0lwsysvj314e0butaj.jpg)

To investigate the effect of MCT on the conv kernels, we again plot the log-power spectra of all the 64 kernels in the first conv layer of the WaveLoc-CONV model.

![image-20211218173012380](https://tva1.sinaimg.cn/large/008i3skNly1gxi48dwbj9j318e0mi45r.jpg)

It can be seen that the first conv layer is now composed of a set of distributed bandpass filters emphasising mainly the 1500-4000Hz range, with some kernels stretching up to 6-7kHz. The low frequencies below 1500Hz are less exploited by the WaveLoc-CONV model.

ILDs are more robust in the high frequency region above 1600Hz. ILD is primarily available at high frequencies.

ITDs are more reliable in the low frequency region below 1600Hz.

It is reasonable to expect that the ITD is more affected by reverberation, while the ILD, created by the head shadowing effect mainly for frequencies higher than 1600Hz, is more robust to reverberation.

Human listeners give ILD more weight than ITD when localising sounds in reverberant conditions.