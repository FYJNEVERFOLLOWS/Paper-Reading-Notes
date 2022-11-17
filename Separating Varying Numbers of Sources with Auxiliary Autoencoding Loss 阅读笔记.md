#! https://zhuanlan.zhihu.com/p/584194851
# Separating Varying Numbers of Sources with Auxiliary Autoencoding Loss 阅读笔记

# Abstract
Many recent source separation systems are designed to separate a fixed number of sources out of a mixture. Iterative methods rely on long-term info to determine the stopping time (not causal); lack a "fault tolerance" mechanism when the estimated number of sources is different from the actual number.

Propose a simple training method, the auxiliary autoencoding permutation invariant training (A2PIT), which assumes a fixed number of outputs and uses auxiliary autoencoding loss to force the invalid outputs to be the copies of the input mixture, and detects invalid outputs in a fully unsupervised way during inference phase.

# 1. Intro
However for a general blind source separation system it is typically not straightforward to obtain the number of active sources in a mixture, especially in inference phase.

A most simple way is to assume a max number of sources in a mixture $N$, and let the model always generate $N$ outputs. For mixtures having $M$ sources when $M \lt N$, $N-M$ outputs are invalid and need to be properly designed and effectively detected. The invalid outputs are typically forced to have a significantly smaller energy than the valid outputs, and a energy threshold can then be applied to filter out those outputs. Drawback: the training targets for the invalid outputs are low- or zero-energy signals, cannot be jointly used with energy-invariant training objectives, such as SI-SDR. Moreover, the detection of invalid outputs typically relies on a pre-defined energy threshold, which may cause trouble when the mixture also has a very low energy.

Another approach first estimates the speaker embedding for each active source with an output-length-free model, e.g. a seq-to-seq generative model, and then performs speaker extraction based on the embeddings. Drawback: spk embedding are typically estimated at utt-level and require a long enough context, which makes the method hard to apply in online or causal systems. Generalization ability on unseen spks is also limited.

The third category of methods perform separation in an iterative way [Listen, think and listen again: capturing top-down auditory attention for speaker-independent speech separation], where in each iteration only one target source is separated from the residual mixture. Drawback: the run-time complexity linearly increases as the number of sources increases, and stop time detection is typically performed at utt-level as well. When there is noise in the mixture, it is also unclear in which iteration should the noise be cancelled.

None of the above methods have a "fault tolerance" mechanism when the estimated number of sources is different than the actual number.

Propose a simple training method based on the fixed-output assumption by designing proper training targets for the invalid outputs. We adopt the fixed-output-number assumption as in real-world conversations such as meeting scenarios, the maximum number of simultaneously active spks is almost always fewer than three. Instead of using low-energy auxiliary targets for invalid outputs, we use the mixture itself as auxiliary targets to force the invalid outputs to perform autoencoding. A2PIT not only allows the model to perform valid output detection in a self-supervised way without additional modules, but also achieves "fault tolerance" by the "*do nothing is better than do wrong things*" principle. As the mixture itself can be treated as the output of a null separation model, i.e., perform no separation at all, the auxiliary targets force the model to generate outputs not worse than doing nothing. Moreover, the detection of invalid outputs in A2PIT can be done at frame-level based on the similarity between the outputs and the mixture, which makes it possible to perform single-pass separation and valid source detection in real-time.

# 2. Auxiliary Autoencoding Permutation Invariant Training
## 2.1. PIT
PIT calculates the loss between the outputs and all possible permutations of the targets, and select the one that corresponds to the minimum loss for back-propagation.

For the problem of separating varying numbers of sources where the actual number of sources are $M \le N, N-M$ auxiliary targets need to be properly designed. A typical way is to use low-energy random Gaussian noise as targets and detect invalid outputs by using a simple energy threshold.

## 2.2. Auxiliary Autoencoding for Invalid Outputs
2 main issues in the energy-based method for invalid output detection.
1. cannot be jointly used with energy-invariant objective functions like SI-SDR.
2. Once the detection of invalid spks fails and the noise signals are selected as the targets, the outputs can be completely uncorrelated with any of the targets, which is unpreferred for applications that require high perceptual quality or low distortion. We define this as the problem of lacking "fault tolerance" mechanism for unsuccessful separation.

We select the mixture signal itself as the auxiliary targets instead of random noise signals. The A2PIT loss:
$$
\mathcal{L}_{obj}=\mathcal{L}_{sep}+\mathcal{L}_{AE}
$$
$\mathcal{L}_{sep}$ for the valid outputs
$\mathcal{L}_{AE}$ is the aux autoencoding loss for the invalid outputs with the input mixtures as targets.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221114215147.png)

## 2.3. Detection of invalid outputs
During inference phase, the detection of invalid outputs can be performed by calculating the similarity, e.g. SI-SDR score, between all outputs and the input mixture, and a threshold calculated from the training set can be used for the decision. For the "fault tolerance" mechanism, the following method is applied for selecting the valid outputs:
1. If estimated number of outputs $K \lt$ the actual number $M$, $M - K$ additional outputs are randomly selected from the $N-K$ remaining outputs.
2. If estimated number of outputs $K \gt$ the actual number $M$, $M$ outputs are randomly selected from the $K$ outputs.

# 3. Exp
## 3.2. Model config
Train DPRNN for each of the speaker count config as the baseline models. These results represents how well the models can achieve when the number of spks is known.

For separating varying numbers of sources, we train DPRNN on three configs:
1. 2+3 spks: 2 and 3 spk mixtures are used for both training and evaluation, and the number of outputs $N$ is set to 3.
2. 2+3+4 spks
3. 1+2+3+4 spks

## 3.4. Results and discussions
### 3.4.1. Determine the similarity threshold through the training set
The similarity threshold in Sec. 2.3 needs to be determined through the training set.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221114221123.png)
Autoencoding SI-SDR: the SI-SDR between outputs and mixture.

Empirically set the threshold for all models for clean separation tasks to be 20, i.e. outputs with autoencoding SI-SDR higher than 20 dB will be treated as invalid outputs. Thresholds for 2+3 model, 2+3+4 model, 1+2+3+4 model for noisy separation tasks to be 12, 12 and 8.

### 3.4.2. Accuracy of speaker counting
### 3.4.3. Performance of speech separation

# 4. Conclusion
Proposed a simple method for separating varying numbers of speakers in a mixture with "fault tolerance" ability — A2PIT.

A2PIT assumed a fixed number of outputs $N$ and appended mixture signals to the training targets of the utts whose number of valid outputs $M$ was smaller than $N$.

Fault tolerance was achieved by treating the auxiliary outputs as the outputs of a "null" separation which directly passed the input to the output. We call this the "do nothing is better than do wrong things" principle. During inference time, a similarity threshold between the mixture and the outputs was used to determine valid outputs in a fully unsupervised way. Experiment results showed that A2PIT was able to effectively perform speaker count in various scenarios, and maintained on par or better separation performance than baseline systems trained for specific datasets with both oracle and predicted speaker count.
