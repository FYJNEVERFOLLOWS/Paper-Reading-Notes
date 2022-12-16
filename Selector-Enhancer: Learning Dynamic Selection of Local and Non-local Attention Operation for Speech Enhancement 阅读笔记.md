# Selector-Enhancer: Learning Dynamic Selection of Local and Non-local Attention Operation for Speech Enhancement 阅读笔记

## [AAAI 2023]
![](https://tva1.sinaimg.cn/large/008vxvgGly1h95ds86eznj31uo0g40xf.jpg)

# Abstract
Natural speech contains many fast-changing and relatively brief acoustic events, therefore, capturing the most informative speech features by indiscriminately using local and non-local attention is challenged. We observe that the noise type and speech feature vary within a sequence of speech and the local and non-local operations can respectively extract different features from corrupted speech. To leverage this, we propose Selector-Enhancer, a dual-attention based convolution neural network (CNN) with a feature-filter that can dynamically select regions from low-resolution speech features and feed them to local or non-local attention operations. In particular, the proposed feature-filter is trained by using reinforcement learning (RL) with a developed difficulty-regulated reward that is related to network performance, model complexity, and “the difficulty of the SE task”. The results show that our method achieves comparable or superior performance to existing approaches. In particular, Selector-Enhancer is potentially effective for real-world denoising, where the number and types of noise are varies on a single noisy mixture.

![](https://tva1.sinaimg.cn/large/008vxvgGly1h95dvsqr4aj31i00u012e.jpg)

![](https://tva1.sinaimg.cn/large/008vxvgGly1h95dwhhj9aj30u00waq6j.jpg)

![](https://tva1.sinaimg.cn/large/008vxvgGly1h95dxqrm1kj31n80g4jvq.jpg)

# Conclusion
Propose Selector-Enhancer that enables attention selection for each speech feature region. Specifically, Selector-Enhancer contains a dual-attention based CNN and a feature filter. The dual-attention based CNN offers options for local and non-local attention mechanisms to address different types of noisy mixtures while the feature-filter selects the optimal regions of speech spectra to feed into the specific attention mechanisms. Comprehensive ablation studies have been carried out, justifying almost every design choice we have made in Selector-Enhancer. Experimental results show that the proposed method is capable of superior performance in SE by outperforming the state-of-the-art methods with fewer computational costs.
