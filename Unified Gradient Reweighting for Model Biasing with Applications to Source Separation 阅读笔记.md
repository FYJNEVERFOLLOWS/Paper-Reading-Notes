# Unified Gradient Reweighting for Model Biasing with Applications to Source Separation 阅读笔记

# Abstract
The vast majority of DL-based separation works is focused on improving average separation performance, often neglecting to examine or control the distribution of the results. Propose a simple, unified gradient reweighting scheme, with a lightweight modification to bias the learning process of a model and steer it towards a certain distribution of results. More specifically, we reweight the gradient updates of each batch, using a user-specified probability distribution. Our framework enables the user to control a robustness trade-off between worst and average performance. 

# 1. Intro
In real-world applications, users are mostly interested towards having a robust system where the failure cases are minimized or might prefer a biased system working best with specific sound classes.

**Bias-variance trade-off**

Larger variance when out of distribution sources appear. 

Curriculum learning starts by using higher weights on "easier" examples while gradually shifting to a uniform weight distribution and has been shown to significantly reduce training time as well as increasing generalization capabilities of models. Reweighting of training examples has also been used for developing more accurate models by focusing on higher variance examples.

We treat the separation model under training as a partially fault tolerant system which is able to continue to operate well in certain cases while failing in others. Biasing estimation models and shifting their operation point in order to make certain examples more significant than others is critical for real-world applications.

2 contributions:
1. Formalize a unified gradient reweighting scheme that can be used by the users in order to shift the operation point of their estimation models for multiple applications.
2. Present a simple way of defining a distribution over the gradients and show the efficacy of our method under a variety of separation tasks and operation modes.

# 2. Unified Gradient Importance Reweighting Scheme for Biasing Estimation Models
## 2.1. Unbiased Estimation Model Training
Parameters update rule:
$$
\boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_k-\eta \sum_{i=1}^B \frac{\mathbf{g}_k^{(i)}}{B}, \mathbf{g}_k^{(i)}=\nabla_{\boldsymbol{\theta}_k} \mathcal{L}\left(f_{\boldsymbol{\theta}}\left(\mathbf{o}^{(i)}\right), \mathbf{s}^{(i)}\right)
$$
$k$: the optimization step

$\eta>0$: the learning rate

$i$: the $i$-th sample in the batch drawn i.i.d. from the dataset $\mathcal{D}$

$\mathbf{\theta}$: parameters

$f_\theta$: estimation model

## 2.2. Unified Gradient Reweighting for Biased Training

$p$: probability mass function (pmf) over all samples in the batch

$$
\begin{gathered}
\widetilde{\boldsymbol{\delta}}_k=\underset{p_k}{\mathbb{E}}\left[\mathbf{g}_k^{(i)}\right]=\sum_{i=1}^B p_k\left(\mathbf{o}^{(i)}, \mathbf{s}^{(i)}\right) \mathbf{g}_k^{(i)} \\
\text { s.t. } \sum_{i=1}^B p_k\left(\mathbf{o}^{(i)}, \mathbf{s}^{(i)}\right)=1, \quad \forall\left\{\left(\mathbf{o}^{(i)}, \mathbf{s}^{(i)}\right)\right\}_{i=1}^B \in \mathcal{D},
\end{gathered}
$$

## 2.3. Softmax Gradient Reweighting
Propose a simple but effective way of parameterizing the pmf distribution using a softmax function.
$$
p_k\left(\mathbf{o}^{(i)}, \mathbf{s}^{(i)}\right)=\frac{\exp \left(\mathrm{F}_k\left(\mathbf{o}^{(i)}, \mathbf{s}^{(i)}\right)\right)}{\left.\sum_{j=1}^B \exp \left(\mathrm{F}_k\left(\mathbf{o}^{(j)}, \mathbf{s}^{(j)}\right)\right)\right)}, \forall i, k,
$$
where $\mathrm{F}$ is a weighting function which will be instantiated according to the operation point that each user needs to shift the estimation model towards.

### 2.3.1. Robust Estimation
In real-world systems, the expected performance is not the only metric that should be taken into consideration. The failure cases of an estimation model might be destructive for downstream tasks (e.g. a bad separation mechanism might lead to deficient performance of an ASR system).
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220919223818.png)

### 2.3.2. Curriculum Training
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220919223936.png)

### 2.3.3. Bias Towards Specific Classes

# 3. Experimental Framework