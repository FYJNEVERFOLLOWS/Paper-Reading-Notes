#! https://zhuanlan.zhihu.com/p/505150122

# Conformer: Convolution-augmented Transformer for Speech Recognition阅读笔记

# Abstract & Conclusion

Transformer models are good at capturing content-based global interactions, while CNNs exploit local features effectively. We combine convolution neural networks and transformers to model both local and global dependencies of an audio sequence.

Conformer, an architecture that intergrates components from CNNs and Transformers for end-to-end speech recognition.

# Intro

While Transformers are good at modeling long-range global context, they are less capable to extract fine-grained local feature patterns. CNNs exploit local information. 

# Conformer Encoder

Our audio encoder first processes the input with a conv subsampling layer and then with a number of conformer blocks.

![image-20220424222040552](https://tva1.sinaimg.cn/large/e6c9d24ely1h1l6bs69b6j20u00ypady.jpg)

![image-20220424230019562](https://tva1.sinaimg.cn/large/e6c9d24ely1h1l7gznrbkj22g40dyadm.jpg)

![image-20220424222259980](https://tva1.sinaimg.cn/large/e6c9d24ely1h1l6e5qnhcj218k0e4gnv.jpg)

# Experiments

## Ablation Studies

![image-20220424230833642](https://tva1.sinaimg.cn/large/e6c9d24ely1h1l7pk4grpj21680u07aw.jpg)