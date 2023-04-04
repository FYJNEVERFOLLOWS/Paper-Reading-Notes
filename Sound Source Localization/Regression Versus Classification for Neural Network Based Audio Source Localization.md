#! https://zhuanlan.zhihu.com/p/619270906
# Regression Versus Classification for Neural Network Based Audio Source Localization 阅读笔记

## WASPAA 2019
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230401181109.png)

# Abstract
Compare the performance of regression and classification neural networks for single-source DOA estimation. Since the output space is continuous and structured, regression seems more appropriate. However, classification on a discrete spherical grid is widely believed to perform better and is predominantly used in the literature. For classification, propose two alternatives to the classical one-hot encoding framework: we derive a Gibbs distribution from the squared angular distance between grid points and use the corresponding probabilities either as soft targets or as cross-entropy weights that retain a clear probabilisitic interpretation.

Regression on Cartesian coordinates is generally more accurate, except when interference is present, in which case classification appears to be more robust.

# 1. Intro
Many classical methods are based on the TDOA between microphones (GCC-PHAT). SRP algorithms build acoustic maps by scanning the space with a beamformer. Subspace algorithms such as MUSIC and ESPRIT use the eigenvalue decomposition of the covariance matrix of the signal to separate the contributions of point sources and diffuse noise. These methods are not able to model the sound scene in real-world situations with reverberation, ambient noise, and where sources are not perfectly punctual, resulting in degraded localization performance.

To overcome the limitations of physical modeling, data-driven approaches propose to use supervised learning in order to grasp the complexity of acoustic phenomena.

In a supervised learning framework, the problem can be formulated either as a regression or as a classification problem. When the output space is not structured and is discrete, e.g. in the case of image recognition, classification is an obvious choice. On the contrary, for problems with a structured and possibly continuous output space, both formulations have assets and drawbacks.

In the context of DOA estimation of audio sources, although the output space is highly structured, most NN based systems rely on multi-label binary classification on the discretized unit sphere. Two notable exceptions can be found: He et al. [Deep Neural Networks for Multiple Speaker Detection and Localization] reintroduced a structure in the output DOA space by likelihood-based encoding of the output of the network, and Adavanne et al. [Sound event localization and detection of overlapping sources using convolutional recurrent neural network] proposed a regression based formulation where the output of the network is formed with the Cartesian coordinates on the unit sphere corresponding to the target DOA.

In this article, we investigate the impact of the framework (classification versus regression) on DOA estimation with NNs. We build on the CRNN architecture designed in our previous work on single-source DOA estimation for Ambisonics recordings. We compare several classification and regression approaches by using targets and loss funcs that are adapted to the geometry of the problem. We notably propose a classification network that accounts for the spherical structure of the output space while enabling a clear probabilistic interpretation.

# 2. Frameworks for the Localization Problem
We clarify the formulations of DOA estimation as regression and classification. We restrict our study to the situation where a single static speaker needs to be localized in a reverberant environment with ambient noise. Only estimate the azimuth and elevation of the spk with respect to the center of the mic array.

## 2.1. Regression
$$
\left\{\begin{array}{l}
\theta \in (-180^{\circ}, 180^{\circ}] \\
\phi \in [-90^{\circ}, 90^{\circ}]
\end{array}\right.
$$
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230401201430.png)

## 2.2. Classification
In the classification formulation, the neural network outputs a score for each class on the discretized unit sphere. The class that is the closest to the actual DOA $(\theta, \phi)$ should get the highest score.

$$
\left\{\begin{array}{l}
\phi_i=-90+\frac{i}{I} \times 180 \quad \text { with } i \in\{0, \ldots, I\} \\
\theta_j^i=-180+\frac{j}{J^i+1} \times 360 \text { with } j \in\left\{0, \ldots, J^i\right\}
\end{array}\right.
$$
where $I=\lfloor\frac{180}{\alpha}\rfloor$ and $J^i=\lfloor\frac{360}{\alpha}\mathrm{cos}\ \phi_i\rfloor$ with $\alpha$ the desired grid resolution in degrees. The resulting grid contains $n_{\mathrm{DOA}}=\sum_{i=0}^I(J^i+1)$

# 3. Proposed Solutions
## 3.1. A shared neuronal basis
Input: [T, F, C]

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230401202707.png)

## 3.2. Configs for Regression
Design 3 NNs to estimate the DOA directly by regression.
**Regression with spherical target:** has two output units with $\theta$ and $\phi$ as targets. MSE is used for the loss.
This loss does not account for the spherical geometry of the outputs.

**Regression with spherical target and angular loss:** identical to the previous one, except that the loss is the angular distance $\delta$ between the prediction $\hat{\psi}=(\hat{\theta},\hat{\phi})$ and the actual DOA $\psi=(\theta, \phi)$:
$$
\begin{aligned}
\delta(\hat{\psi}, \psi)=\arccos \{ & \sin (\hat{\phi}) \sin (\phi) \\
& +\cos (\hat{\phi}) \cos (\phi) \cos (\hat{\theta}-\theta)\}
\end{aligned} \tag{3}
$$
This loss accounts for the spherical geometry of the output space. However, mapping from directions on the unit sphere to azimuth and elevation coordinates is unstable near the poles.

**Regression with Cartesian target:** in order to solve the latter issue, design a network that targets the 3 Cartesian coordinates of the unit vector pointing towards the DOA, as in [Sound event localization and detection of overlapping sources using convolutional recurrent neural network]. MSE is used for the loss. In this case, it actually represents a geometrical distance between the prediction and the true DOA.

For all these networks, the labels are scaled between 0 and 1 and a sigmoid is applied on the last layer. As a post-processing in the prediction step, we average the outputs on all frames of the sequence, in order to return one prediction per sequence.

## 3.3. Configs for Classification
**CE with one-hot encoded target:** softmax at the output layer and optimize a CE loss. The target distribution is taken as the one-hot in the grid point which is closest to the actual DOA.

**MSE with soft Gibbs target:** we induce some structure on the output space with a softer target: a Gibbs distribution with energy taken as the angular distance between grid points $\psi{i,j}$ and the true DOA $\psi$, similarly to [Deep Neural Networks for Multiple Speaker Detection and Localization]:

$$
\mathcal{G}\left(\psi_{i j}\right)=e^{-\delta\left[\psi_{i j}, \psi\right]^2 / \beta^2}
$$
where $\delta$ is the angular distance (3) and $\beta$ defines an angular neighborhood. Use sigmoid in the last layer and MSE loss. The interpretability of the output as a probability distribution is thus lost.

**Gibbs-weighted loss:** to keep the structure of the ouput space while allowing a clear probabilistic interpretation, we integrate the Gibbs distribution in the CE loss using cost-sensitive weights:
$$
\operatorname{loss}=-\log \left(\sigma_{i j}\right)-\sum_{\substack{\left(i^{\prime}, j^{\prime}\right) \\ \neq(i, j)}}\left(1-\mathcal{G}\left(\psi_{i^{\prime} j^{\prime}}\right)\right) \log \left(1-\sigma_{i^{\prime} j^{\prime}}\right)
$$
where $\psi_{ij}$ is the grid point that is the closest to the actual DOA and $\sigma_{ij}$ is the output of the network for the class $\psi_{ij}$ after a sigmoid. 

In the prediction step, as for regression, the outputs $\sigma_{ij}$ of the network are averaged over all frames of a seq for all grid points. The estimated DOA then corresponds to the class with the highest global score.

# 4. Exp
$\alpha = 10^{\circ}, n_{\mathrm{DOA}}=429$

Nadam

$\beta=2\alpha=20^{\circ}$

# 5. Results
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202304/20230401211654.png)
For all test sets, classification with a Gibbs distribution as the target is slightly better than with a one-hot target. Using Gibbs-weighted loss further improves the performance, especially for real recordings where the number of outliers significantly decreases. It additionally speeds up the training, with only 60 epochs needed instead of 100 with a comparable computation time per epoch.

# 6. Conclusion
No single system stands out. The results are tightly linked to the evaluation scenario and metric.


Regression appears to be more accurate in scenarios with diffuse interference.

On the contrary, classification seems more robust to localized interference.