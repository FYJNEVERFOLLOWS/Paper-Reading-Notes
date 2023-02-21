#! https://zhuanlan.zhihu.com/p/592274391
# Unsupervised Sound Separation Using Mixture Invariant Training 阅读笔记
## [NIPS 2020]
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202212/20221201214021.png)
# Abstract
A separation model is trained to predict the component sources from synthetic mixtures created by adding up isolated ground-truth sources. Reliance on this synthetic training data is problematic because good performance depends upon the degree of match between the training data and real-world audio, especially in terms of the acoustic conditions and distribution of sources. The acoustic properties can be challenging to accurately simulate, and the distribution of sound types may be hard to replicate.

Propose a completely unsupervised method, mixture invariant training (MixIT), that requires only single-channel acoustic mixtures. In MixIT, training examples are constructed by mixing together existing mixtures, and the model separates them into a variable number of latent sources, such that the separated sources can be remixed to approximate the original mixtures. We show that MixIT can achieve competitive performance compared to supervised methods on speech separation. Using MixIT in a semi-supervised learning setting enables unsupervised domain adaptation and learning from large amounts of real world data without ground-truth source waveforms.

# 1. Intro
**Good survey, worth reading**
Individual sounds are convolved with unknown acoustic reverberation functions and mixed together at the acoustic sensor in a way that is impossible to disentangle without prior knowledge of the source characteristics.

PIT explicitly outputs the signals in an arbitrary order, and the loss function finds the permutation of that order that best matches the estimated signals to the references. In both DPCL and PIT, the ground-truth signals are inherently part of the loss.

A major problem with supervised training for source separation is that it is not feasible to record both the mixture signal and the individual ground-truth source signals in an acoustic environment. Therefore supervised training has relied on synthetic mixtures created by adding up isolated ground-truth sources, with or without a simulation of the acoustic environment.

It is difficult to match the characteristics of a real dataset because the distribution of source types and room characteristics may be unknown and difficult to estimate, data of every source type in isolation may not be readily available, and accurately simulating realistic acoustics is challenging.

One approach to avoiding these difficulties is to use acoustic mixtures from the target domain, without references, directly in training. To that end, weakly supervised training has been proposed to substitute the strong labels of source references with another modality such as class labels [Finding strength in weakness: Learning to separate sounds with weak supervision], visual features, or spatial info [Bootstrapping single-channel source separation via
unsupervised spatial clustering on stereo mixtures, Unsupervised deep clustering for source separation: Direct learning from mixtures using spatial information]. 

Propose *mixture invariant training* (MixIT), a novel unsupervised training framework that requires only single-channel acoustic mixtures, which generalizes PIT in that the permutation used to match source estimates to source references is relaxed to allow summation over some of the sources. Instead of single-source refs, MixIT uses mixtures from the target domain as refs, and the input to the separation model is formed by summing together these mixtures to form a mixture of mixtures. The model is trained to separate this input into a variable number of latent sources, such that the separated sources can be remixed to approximate the original mixtures.

**Contributions:**
(1) propose the first purely unsupervised learning method for audio-only single-channel separation

(2) provide extensive experiments with cross-domain adaptation to show the effectiveness of MixIT for domain adaptation to diff reverberation characteristics in semi-supervised settings

(3) opens up the use of a wider variety of data, such as training speech enhancement models from noisy mixtures by only using speech activity labels, or improving performance of universal sound separation models by training on large amount of unlabeled, in-the-wild data.

# 2. Relation to previous work
Reference worth reading.

Unlike previous supervised approaches, MixIT can use a database of only mixtures as references, enabling training directly on target-domain mixtures for which ground-truth source signals cannot be obtained.

*adversarial unmix-and-remix*

Recent works to such unsupervised domain adaptation (adapt to domains for which it is difficult to obtain ref source signals) have used adversarial training to learn domain-invariant intermediate network activations, learn to translate synthetic inputs to the target domain, or train student and teacher models to predict consistent separated estimates from supervised and unsupervised mixtures [Mixup-breakdown: a consistency training method for improving generalization of speech separation models].

In contrast, we take a semi-supervised learning approach and jointly train the same network using both supervised and unsupervised losses, w/o making explicit use of domain labels.

# 3. Method
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221129221514.png)

Assume that the max number of sources which may be present in the mixtures is known.

## 3.1. PIT
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202211/20221130110148.png)
## 3.2. MixIT
Main limitation of PIT: it requires knowledge of the ground truth source signals $\mathbf{s}$, and therefore cannot directly leverage unsupervised data where only mixtures $x$ are observed. MixIT overcomes this problem as follows. Two mixtures $x_1$ and $x_2$, each comprised of up to $N$ underlying sources, MoM: $\overline{x}=x_1+x_2$. Separation model predicts $M \ge 2N$ source signals.

$$
\mathcal{L}_{\text {MixIT }}\left(x_1, x_2, \hat{\mathbf{s}}\right)=\min _{\mathbf{A}} \sum_{i=1}^2 \mathcal{L}\left(x_i,[\mathbf{A} \hat{\mathbf{s}}]_i\right), \tag{2}
$$

where $\mathcal{L}$ is the same signal-level loss used in PIT (2) and the $mixing matrix$ $\mathbf{A} \in \mathbb{B}^{2 \times M}$ is constrained to the set of $2 \times M$ binary matrices where each column sums to 1, i.e. the set of matrices which assign each source $\hat{s}_m$ to either $x_1$ or $x_2$. MixIT minimizes the total loss between mixtures $\mathbf{x}$ and remixed separated sources $\hat{\mathbf{x}} = \mathbf{A} \hat{\mathbf{s}}$.

## 3.3. Semi-supervised training
Each training batch contains $p$% supervised data, for which we use the PIT loss, and the remaining contains unsupervised mixtures, for which we do not know their constituent sources and use the MixIT loss.

# 4. Exp

## 4.1. Speech separation
Try two variants of this task: mixtures that always contain two speakers (2-source) such that MoMs always contain four sources, and mixtures containing either one or two speakers (1-or-2-source) such that MoMs contain two to four sources. The network always has four outputs.

## 4.2. Speech enhancement