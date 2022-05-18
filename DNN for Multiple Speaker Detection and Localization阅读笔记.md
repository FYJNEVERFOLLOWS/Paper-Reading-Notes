#! https://zhuanlan.zhihu.com/p/413083178
# DNN for Multiple Speaker Detection and Localization阅读笔记
# Introduction

SSL and speaker detection are crucial components in human-robot interaction, where the robot needs to precisely detect where and who the speaker is and responds appropriately.

Challenges:

Noisy environments; Multiple simultaneous speakers; Short and low-energy utterances; Obstacles such as robot body blocking sound direct path.

Conventional signal processing algorithms are derived with assumptions, many of which do not hold well under the above-mentioned conditions.

NNs can learn the mapping from the localizaition cues to the direction-of-arrival without making strong assumptions.

Motivation: (Compared to Existing NN-based SSL Methods)

Do not address the problem of multiple sound sources;

Cannot detect and localize multiple voices in real multi-party HRI scenarios simultaneously.

Formulate the problem as the classification of an audio input into one "class" label associated with a location, and optimizing the posterior probability of such labels.Such posterior probability encoding cannot be easily extended to multiple sound source situations.

![image-20210919112124972](https://tva1.sinaimg.cn/large/008i3skNly1gulrsxx4pwj60vd07tdhi02.jpg)

Contributions:

Our methods can cope with short input, overlapping speech, an unknown number of sources and strong ego-noise.

# Proposed Method

time frame: 170ms

number of sources: $N$

number of microphones: $M$

STFT of input signal: $X_i(\omega),i=1,...,M$, the mic index: $i$, the discrete freq: $\omega$.



A. Input Features

GCC-PHAT is an import clue for SSL.

Two types of features based on GCC-PHAT at frame level:

* GCC-PHAT coefficients: 

The GCC-PHAT between channel $i$ And $j$ Is formulated as:
$$
g_{i j}(\tau)=\sum_{\omega} \mathcal{R}\left(\frac{X_{i}(\omega) X_{j}(\omega)^{*}}{\left|X_{i}(\omega) X_{j}(\omega)^{*}\right|} e^{j \omega \tau}\right)
$$
$\tau$$(\in[-25,25])$ is the discrete delay (use the center 51 delays), (·)* denotes the complex conjugation, $\mathcal{R}(·)$ denotes the real part of a complex number. 

带有噪音和混响的真实环境下，the full GCC-PHAT function is used as the input feature instead of the peak in GCC-PHAT.

* GCC-PHAT on Mel-scale filter bank: 

The GCC-PHAT is not optimal for TDOA estimation of multiple source signals since it equally sums over all freq bins disregarding the "sparsity" of speech signals in the TF domain. [The Mel-scale aims to mimic the non-linear human ear perception of sound, by being more discriminative at lower frequencies and less discriminative at higher frequencies.] 所以提出 GCC-PHAT on Mel-scale filter bank (GCCFB).
$$
g_{i j}(f, \tau)=\frac{\sum_{\omega \in \Omega_{f}} \mathcal{R}\left(H_{f}(\omega) \frac{X_{i}(\omega) X_{j}(\omega)^{*}}{\left|X_{i}(\omega) X_{j}(\omega)^{*}\right|} e^{j \omega \tau}\right)}{\sum_{\omega \in \Omega_{f}} H_{f}(\omega)}
$$
$f$ is the filter index, $H_f$ is the transfer function of the f-th Mel-scaled triangular filter, and $\Omega_f$ is the support of $H_f$.

![image-20210918144810886](https://tva1.sinaimg.cn/large/008i3skNly1guks65tdq6j60ns0ir40g02.jpg)

40 Mel-scale filters covering the frequencies from 100 to 8000 Hz.



B. Likelihood-based Output Coding

**Encoding:** the output (the likelihood of a sound source being in each direction) is encoded into a vector {$o_i$} of 360 values (分别对应 $\theta_i$ ). The values are defined as the maximum of Gaussian-like functions centered around the true DOAs:
$$
o_{i}= \begin{cases}\max _{j=1}^{N}\left\{e^{-d\left(\theta_{i}, \theta_{j}^{(s)}\right)^{2} / \sigma^{2}}\right\} & \text { if } N>0 \\ 0 & \text { otherwise }\end{cases}
$$
其中 $\theta_j^{(s)}$ 是第 j 个声源DOA的真实值，$\sigma$ 是高斯分布的标准差（尺度参数）[ beam width: 3.1-3.4°]，$d(·,·)$ 表示 angular distance

![image-20210918165932712](https://tva1.sinaimg.cn/large/008i3skNly1gukvyfbhf8j60mo07hq3j02.jpg) 

The output coding 会在 true DOAs / true azimuth angle出现 peaks

Posterior probability coding is constrained as a probability distribution (the output layer is normalized by a softmax function). It can be all zero when there is no sound source, or contains N peaks when there are N sources. 就不像概率分布函数那样，不管各个方向概率大小，求和总是为1。The coding can handle the detection of an arbitrary number of sources.（极端的假设，当声源无限多时，概率分布在各方向上都很小，可能就无法区分有声源的方向和无声源的方向）。

**Decoding:** During the test phase, we decode the output by finding the peaks that are above a given threshold $\xi$ :
$$
\text { Prediction }=\left\{\theta_{i}: o_{i}>\xi \quad \text { and } \quad o_{i}=\max _{d\left(\theta_{j}, \theta_{i}\right)<\sigma_{n}} o_{j}\right\}
$$
with $\sigma_n$ being the neighborhood distance. We choose $\sigma = \sigma_n = 8°$ for the experiments.



C. 3 different Neural Network Architectures

**MLP-GCC (Multilayer perceptron with GCC-PHAT)**

![image-20210918215538703](https://tva1.sinaimg.cn/large/008i3skNly1gul4ikd7fkj60n60gaq4j02.jpg)

Three hidden fully-connected layers with ReLU activation function and Batch Normalization

The last fully-connected layer with sigmoid activation function



**CNN-GCCFB (Convolutional neural network with GCCFB)**

FC NNs are not suitable for high-dimensional input features (such as GCCFB) (introduces a large amount of parameters; prone to overfitting).

Four convolutional layers (with ReLU activation and BN) and a FC layer at the output (with sigmoid activation)



**TSNN-GCCFB (Two-stage neural network with GCCFB)**

CNN-GCCFB considers the input features as images without taking their properties into account. We design the weight sharing in the network with the knowledge about the GCCFB for the third architecture.

* do analysis or implicit DOA estimation in each freq band before such info is aggregated into a broadband prediction
* Features with the same delay on different microphone pairs do not correspond to each other locally. Instead, feature extraction or filters should take the whole delay axis into account.

The first stage extracts latent DOA features in each filter bank, by repeatedly applying Subnet 1 (2-hidden-layer MLP) on individual frequency regions that span all delays and all microphone pairs.

![image-20210918221812100](https://tva1.sinaimg.cn/large/008i3skNly1gul55zr3hxj60nt0h9gny02.jpg)

The second stage aggregates information across all frequencies in a neighbor DOA area and outputs the likelihood of a sound being in each DOA. Similarly, the Subnet 2 (1-hidden-layer MLP) is repeatedly used for all DOAs in the second stage. All the hidden layers are of size 500.

Training scheme: First, we train the Subnet 1 in the first stage using the DOA likelihood as the desired latent feature (局部特征). During the second step, both stages are trained in an end-to-end manner.

# Experiment

A. Datasets

![image-20210918225641139](https://tva1.sinaimg.cn/large/008i3skNly1gul6a3bp2xj60n40d7q4h02.jpg)

sr: 48kHz

4 mics, rectangle of 5.8 X 6.9 cm



B. Evaluation Protocol

We evaluate multiple SSL methods at frame level under two different conditions: the number of sources is known or unknown.

Known: We select the N highest peaks of the output as the predicted DOAs and match them with ground truth DOAs one by one, and we compute the mean absolute error (MAE). Evaluate by ACC of predictions.

By saying a prediction is correct, we mean the error of the prediction is less than a given admissible error $E_a$.

Unknown: 

detection - given ground truth sources, compute recall (the percentage of correct detection out of all ground truth sources)

localization - compute precision (the percentage of correct predictions among all predictions)



C. Network Training

Adam optimizer

MSE loss

Mini-batch size: 256

10 epochs for MLP-GCC & CNN-GCCFB

4 epochs for the first stage of TSNN-GCCFB and 10 epochs for the end-to-end training



D. Baseline Methods

Spatial spectrum-based methods:

SRP-PHAT

SRP-NONLIN: SRP-PHAT with a non-linear modification of the score

MVDR-SNR: minimum variance distortionless response beam forming with signal-to-noise ratio as score

SEVD-MUSIC: multiple signal classification, assuming spatially white noise and one signal in each bin

GEVD-MUSIC: MUSIC with generalized eigenvector decomposition, assuming noise is pre-measured and one signal in each TF bin



E. Results

![image-20211104092210707](https://tva1.sinaimg.cn/large/008i3skNly1gw2uv1x68nj31780d6425.jpg)

Note that, the loudspeaker dataset is in general more challenging because it contains samples with lower SNR and wider range of azimuth directions.

![image-20210919095540852](https://tva1.sinaimg.cn/large/008i3skNly1gulpbp9xe0j60tw0mon2202.jpg)



# Conclusion

Limitation: The current study is potentially limited by the training data samples, which are not likely to cover all possible combinations of source positions, since the number of combinations grows exponentially with the number of sources. Will investigate the incorporation of temporal context.

