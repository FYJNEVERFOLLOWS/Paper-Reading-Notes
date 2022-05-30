#! https://zhuanlan.zhihu.com/p/519795224
# [时域语音分离] Sudo rm -rf: Efficient Networks for Universal Audio Source Separation 阅读笔记

# Abstract
Present an efficient neural network for end-to-end general purpose audio source separation. The backbone structure of this convolutional network is the SUccessive DOwnsampling and Resampling of Multi-Resolution Features as well as their aggregation which is performed through simple one-dimensional convolutions.

# Intro
Previous methods have high computational complexity.

Our model performs SUccessive DOwnsampling and Resampling of Multi-Resolution Features using depth-wise convolutions.

SuDoRM-RF models a) could be deployed on devices with limited resources, b) could be trained significantly faster and achieve good separation performance and c) scale well when increasing the number of parameters.

# Network Architecture
SuDoRM-RF performs end-to-end audio source separation using a mask-based architecture with adaptive encoder and decoder basis.

$x$: input mixture

$\varepsilon$: encoder

$\mathbf{v}_\mathbf{x}=\varepsilon(\mathbf{x})$: latent representation obtained by encoder

The latent mixture representation is fed through the separation module $S$ which estimates the corresponding masks $\hat{\mathbf{m}}_i \in \mathbb{R}^{C_\varepsilon \times L}$ for each one of the $N$ sources $\mathbf{s}_1,\mathbf{s}_2,...,\mathbf{s}_N$. The estimated latent representation for each source in the latent space $\hat{\mathbf{v}}_i$ is retrieved by multiplying element-wise an estimated mask $\hat{\mathbf{m}}_i$ with the encoded mixture representation $\mathbf{v}_{\mathbf{x}}$. The reconstruction for each source $\hat{\mathbf{s}}_i$ is obtained by $\hat{\mathbf{s}}_i=D(\hat{\mathbf{v}}_i)$.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202205/20220523172434.png)

In $Conv1D_{C,K,S}$ and $ConvTr1D_{C,K,S}$, $C$ for output channels, $K$ for kernel size, $S$ for stride.

## 2.1. Encoder

The encoded input mixture representation can be expressed as:
$$
\mathbf{v}_{\mathbf{x}}=\mathcal{E}(\mathbf{x})=\operatorname{ReLU}\left(\operatorname{Conv} 1 \mathrm{D}_{C_{\mathcal{E}}, K_{\mathcal{E}}, K_{\mathcal{E}} / 2}(\mathbf{x})\right) \in \mathbb{R}^{C_{\mathcal{E}} \times L}
$$
$L$ for temporal axis.

## 2.2. Separator
The separator module $S$ performs the following transformations to the encoded mixture representation $\mathbf{v}_{\mathbf{x}}$:
1. Projects the encoded mixture representation $\mathbf{v}_{\mathbf{x}}$ to a new channel space through a LN followed by a point-wise convolution as shown next:
$$
\mathbf{y}_{0}=\operatorname{Conv} 1 \mathrm{D}_{C, 1,1}\left(\mathrm{LN}\left(\mathbf{v}_{\mathbf{x}}\right)\right) \in \mathbb{R}^{C \times L}
$$
2. Performs repetitive non-linear transformations provided by $B$ U-ConvBlocks on the intermediate representation $\mathbf{y}_{0}$. Denote the output of the $i$-th U-ConvBlock as $\mathbf{y}_{i}$.
3. Aggregates the info over multiple channels by applying a regular one-dimensional convolution for each source on the transposed feature representation $\mathbf{y}_{B}^T$. Obtain an intermediate latent representation for the $i$-th source as shown next:
$$
\mathbf{z}_{i}=\operatorname{Conv} 1 \mathrm{D}_{C, C_{\mathcal{E}}, 1}\left(\mathbf{y}_{B}^{T}\right)^{T} \in \mathbb{R}^{C_{\mathcal{E}} \times L}
$$   
4. Performe a softmax to get mask estimates which add up to one across the dimension of the sources. The mask estimate for $i$-th source would be:
$$
\widehat{\mathbf{m}}_{i}=\operatorname{vec}^{-1}\left(\frac{\exp \left(\operatorname{vec}\left(\mathbf{z}_{\mathrm{i}}\right)\right)}{\sum_{j=1}^{N} \exp \left(\operatorname{vec}\left(\mathbf{z}_{j}\right)\right)}\right) \in \mathbb{R}^{C \mathcal{E} \times L}
$$  
$\text{vec}(\cdot)$ and $\text{vec}^{-1}(\cdot)$ denote the vectorization and the inverse operation.

5. Elementwise the encoded mixture presentation $\mathbf{v}_\mathbf{x}$ and the corresponding mask $\widehat{\mathbf{m}}_{i}$.
$$
\widehat{\mathbf{v}}_{i}=\mathbf{v}_{\mathbf{x}} \odot \widehat{\mathbf{m}}_{i} \in \mathbb{R}^{C \mathcal{E} \times L},
$$

## 2.3. Decoder
$$
\widehat{\mathbf{s}}_{i}=\mathcal{D}_{i}\left(\widehat{\mathbf{v}}_{i}\right)=\operatorname{ConvTr} 1 \mathrm{D}_{C_{\mathcal{E}}, K_{\mathcal{E}}, K_{\mathcal{E}} / 2}\left(\widehat{\mathbf{v}}_{i}\right)
$$

# Experimental Setup

## 3.2. Data preprocessing and generation
Follow the same data aug process in [Efthymios Tzinis, Shrikant Venkataramani, Zhepei Wang, Cem Subakan, and Paris Smaragdis, “Two-step sound source separation: Training on learned latent targets,” in Proc. ICASSP, 2020.]:

A) randomly choosing two sound classes or speakers
B) randomly cropping of 4 sec segments from two sources audio files
C) mixing the source segments with a random SIR

For each epoch, 20,000 new training mixtures are generated. Val and test sets are generated once with each one containing 3,000 mixtures. Downsample each audio clip to 8kHz, subtract its mean and divide with the standard deviation of the mixture.

## 3.3. Training and evaluation details
120 epochs; batch size: 4; SI-SDR loss; SI-SDRi for evaluation

## 3.4. SuDoRM-RF configurations

For the encoder and decoder modules use a kernel size $K_{\varepsilon}=21$ and a number of basis equal to $C_{\varepsilon}=512$.

For the configuration of each U-ConvBlock, the number of input channels $C=128$, the number of successive resampling operations $Q=4$ and the expanded number of channels equal to $C_U=512$.

SuDoRM-RF 1.0x, SuDoRM-RF 0.5x, SuDoRM-RF 0.25x consist of 16, 8 and 4 U-ConvBlocks, respectively.

Adam optimizer with an initial lr set to 0.001 and we decrease it by a factor of 0.2 every 50 epochs.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202205/20220525160205.png)

# Conclusions

The proposed model is capable of extracting multi-resolution temporal features through successive depth-wise convolutional downsampling of intermediate representations and aggregates them using a non-parametric interpolation scheme.