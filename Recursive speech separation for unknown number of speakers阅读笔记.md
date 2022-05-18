#! https://zhuanlan.zhihu.com/p/447497478
# Recursive speech separation for unknown number of speakers阅读笔记
# 0. Abstract & Conclusion

Propose a method of single-channel speaker-independent multi-speaker speech separation for an unknown number of speakers.

OR-PIT: one-and-rest permutation invariant training

Deal with different numbers of speakers cases using a single model.

# 1. Introduction

Previous methods output the maximum number of speakers regardless of the actual number of speakers. M - N are enforced to output silent signals.



To address this problem, we propose to progressively separate speeches by applying a speech separation network recursively. Instead of separating all speakers in a mixture at once, the proposed model separates only one speaker from a mixture at a time and the residual signal is fed back to the separation model for the recursion to separate the next speaker.

![image-20211215141046375](https://tva1.sinaimg.cn/large/008i3skNly1gxehm0ebv6j31v60kidk6.jpg)

We further propose a method of robustly determining when to stop the iteration for an unknown number of speakers.

Another advantage of the proposed method is that it tends to separate first a speaker that is easy to separate and sequentially tackle those that are harder to separate.

Contributions:

1. Propose a recursive speech separation method for separating a mixture of different numbers of speakers with a single model, even for mixtures which have more number of speakers than the mixtures used for training. To train the recursive separation model, we propose OR-PIT.
2. Propose a recursion stopping method that enables to operate the recursive speech separation model for an unknown number of speakers.
3. The proposed model can work well for a four-speaker mixture, which is greater than the number of speakers in the mixtures used for training.
4. Our approach can more accurately detect the number of speakers than the naive approach of directly classifying the number of speakers.

# 2. Recursive Speech Separation

## 2.1. Recursive speech separation

For the j-th recursion step:
$$
\hat{s}^{j}(t), \hat{r}^{j}(t)=F\left(\hat{r}^{j-1}(t)\right)
$$
$\hat{r}^{j}(t)$: the mixture of residual speaker sources, consists of $N - j$ speakers, $\hat{r}^{0}(t)=x(t)$

$\hat{s}^{j}(t)$: one speaker source

F(): the recursive speech separator (neural network)

## 2.2. One-and-Rest PIT

There are $N$ valid choices of one target speaker $s_i(t)$ and corresponding residual signal.

Random or constant choice of the target speaker fails since we do not assume any prior on the order of sources, and the model becomes confused how to choose the speaker during the test time.

To address this problem, we propose novel training method called OR-PIT inspired by uPIT.

OR-PIT computes the error $l$ between the network output and the target for $N$ possible splits of one and rest assignment, $s_i(t),r_i(t)$. The assignment that yields the lowest loss is used for the training objective $L$ to optimize the network,
$$
L=\min _{i} l\left(\hat{s}(t), s_{i}(t)\right)+\frac{1}{N-1} l\left(\hat{r}(t), \sum_{n \neq i} s_{n}(t)\right)
$$
SI-SNR (scale-invariant signal-to-noise ratio):
$$
\left\{\begin{array}{l}
s_{\text {target }}:=\frac{\langle\hat{s}, s\rangle s}{\|s\|^{2}} \\
e_{\text {noise }}:=\hat{s}-s_{\text {target }} \\
l_{\mathrm{SI}-\mathrm{SNR}}(\hat{s}, s):=10 \log _{10} \frac{\left\|s_{\text {target }}\right\|^{2}}{\left\|e_{\text {noise }}\right\|^{2}}
\end{array}\right.
$$
$\hat{s}$ and $s$ are the mean normalized estimates and targets.

## 2.3. Iteration termination criteria

Propose a simple deep neural network based binary classifier that accepts the residual outputs $\hat{r}^j$, and predicts whether the signal is speech or not at each recursion step. If $\hat{r}^j$ is not speech, we stop the recursion and estimate $N$ as $j$.

# 3. Experiments

## 3.1. Network training

Randomly select 2 or 3 utterances of different speakers from WSJ0 and mixing them at random SNR between -2.5 dB and 2.5 dB. 8kHz.

Adopted TASNet as the network architecture. Softmax -> ReLU

First network output always has one speaker, second output channel to collect all the remaining speakers in the mixture input.

10e-3, 100 eps, Adam with a weight decay of 10e-5, input mixtures were 4 secs long with 50% overlap

fine tune:

the model was fine tuned on a 2-speaker mixture obtained from a separation of the first iteration of 3-speaker mixture, instead of clean 2-speaker mixture. The loss was accumulated on both the first iteration (clean 3-speaker separation) and the second iteration (residual 2-speaker separation), and back-propagated.

![image-20211215162705118](https://tva1.sinaimg.cn/large/008i3skNly1gxeljrgx1aj31700egmza.jpg)

Improve the performance of the 3-speaker separation while slightly decrease the performance of 2-speaker separation.

## 3.2. Comparison with other approaches

![image-20211215163525249](https://tva1.sinaimg.cn/large/008i3skNly1gxelsfwfe0j31u00u011w.jpg)

2/3-speaker model: trained for the 2/3-speaker separation task

2&3-speaker model: trained and can be applied to both 2- and 3-speaker separation tasks with a unified model

## 3.3. Identification of number of speakers

Need to know when to terminate the recursion.

Trained the Alexnet model for the task of binary classification of speech or noise on the residual inputs coming out from the second channel.

![image-20211215164231824](https://tva1.sinaimg.cn/large/008i3skNly1gxelzu9rnrj316c0d8jtg.jpg)

Multi-class classifier as baseline: the Alexnet model counting the number of speakers in the input mixture.

## 3.4. Dominant speech separation

A notable property of the proposed method is that it tends to separate the most dominant (easiest) speaker first and successively tackles the separation of less dominant (harder) speakers.

![image-20211215165108204](https://tva1.sinaimg.cn/large/008i3skNly1gxem8saihkj316m0oajui.jpg)

