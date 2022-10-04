#! https://zhuanlan.zhihu.com/p/570495059
# Mining Hard Samples Locally And Globally For Improved Speech Separation 阅读笔记

# Abstract
Speech separation dataset typically consists of hard and non-hard samples, and the former is minority and latter majority. The data imbalance problem biases the model towards non-hard samples and weakens the generalization capability. Given that the average separation performance is sufficiently good, improving hard samples may contribute more to back-end tasks.

Propose two methods to alleviate data imbalance in speech separation task, based on local and global hard sample mining. For the local, we propose weighted loss to compensate for hard samples by increasing their weights in each batch. For the global, we perform global hard sample mining and re-sample to increase the proportion of hard samples in the training set.

Because hard sample mining using objective loss in dynamic mixing leads to local results, propose an indirect method using speaker-specific parameters, based on the fact that pitch median difference and x-vector cosine distance of two speakers in a mixture are closely correlated with separation SI-SNRi.


# 1. Introduction
In the training process, the data are sampled uniformly. Therefore, unbiased models are trained with large variances, which leads to more failures, i.e., hard samples, in the generalization. Given that the average separation performance is sufficiently good, improving hard samples may contribute more to back-end tasks (e.g., speech recognition).

According to objective loss, training samples with low performance can be regarded as hard and the others as non-hard. Typically, the former is the minority and latter the majority.

Assume that uniform sampling in the training set leads to data imbalance, and balancing between hard and non-hard samples may benefit generalization.

Previous studies on data imbalance mainly focus on classification tasks. Solutions include re-sampling, cost-sensitive weighting, re-weighting and data augmentation. **[ Paper list: Learning deep representation for imbalanced classification, Class rectification hard mining for imbalanced deep learning, Learning imbalanced datasets with label-distribution-aware margin loss, Rethinking the value of labels for improving class-imbalanced learning ]**. Data imbalance in regression tasks has not been as well explored [ Delving into deep imbalanced regression ].

Propose two methods from the perspective of hard sample mining to alleviate data imbalance, both of which are based on dynamic mixing [ Wavesplit: End-to-end speech separation by speaker clustering ]. For local hard samples mining, we search hard samples using objective loss in each batch and reweight to compensate for hard samples by increasing their weights during training. We also apply the weighted loss in the validation stage to select the model biased towards hard samples. However, local hard samples may lead to sub-optimal results.

Propose an indirect method for global hard sample mining. Specifically, we first analyze the correlation between speaker-specific parameters (pitch median difference of two spks and the x-vector cosine distance of two spks) and the separation results.

The method based on global hard sample mining shows more promising results than the local method.

Contributions:
1. Propose a novel weighted loss based on local hard sample mining.
2. Discover that the x-vector cosine distance between two spks in a mixture is correlated with the separation results.
3. Propose a novel indirect method for global hard sample mining and a new data augmentation method using hard re-sampling.

# 2. Dynamic mixing based on global hard sample mining
The proposed method consists of three steps: dynamic mixing, hard sample mining and hard re-sampling, as shown in Fig. 1. Specifically, we search hard samples globally in the training set generated using dynamic mixing. Instead of using objective loss, the search is performed using speaker-specific parameters indirectly, based on the correlation between these parameters and the separation results. Then we re-sample to increase the proportion of hard samples.
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20220925171523.png)

## 2.1. Dynamic mixing
[ Wavesplit: End-to-end speech separation by speaker clustering ]

## 2.2. Correlation analysis
The fundamental frequency, i.e., pitch, denoted by $f_0$, is an intrinsic property of periodic signals.

The x-vectors, which map variable-length utts to fixed-dimensional embeddings, capture speaker characteristics. The cosine distance of x-vectors is typically used to discriminate between speakers.

Pearson's correlation coefficient is used to measure the correlation between speaker-specific parameters and separation results.

## 2.3. Hard sample mining
## 2.4. Hard re-sample

# 3. Weighed loss based on local hard sample mining
Propose an alternative method based on local hard sample mining. SI-SNR biases the trained model towards non-hard samples in the majority.
$$
\text{wSI-SNR} = \sum_{i=1}^{B}{w_i}{l_i}
$$
$B$ for batch size

$l_i$ is the SI-SNR for the $i$th audio pair

$w_i$ is the new weight

Sort $l_i$ in descending order in a batch and determine $w_i$ using its index:
$$
w_i=\frac{i}{\sum^B_{i=1}i}
$$
where $i$ is the index after sorting. It is worth noting that the weighted loss compensates for hard samples by increasing their weights in the objective loss, while dynamic mixing with hard sample mining compensates for hard samples by increasing their proportion in the training set. They achieve the similar goal from different aspects.

Also apply wSI-SNR during validation (calculate weights over all validation samples instead of one batch). The weighted validation loss helps to select models that are more biased towards hard samples.

# 5. Results
![Dont understand this fig, plz leave a comment if you know](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20221004092358.png)

## 5.2. Results of speech separation
DM: dynamic mixing

WTL: weighted loss for training set

WVL: weighted loss for validation set
![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20221004092647.png)

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202209/20221004092743.png)

# 6. Conclusions
Explore an improved separation model for hard samples instead of training a model using average metrics. We assume that sampling uniformly in the training set leads to data imbalance. Local hard sample mining: weighted loss and global hard sample mining: dynamic mixing with hard sample mining.