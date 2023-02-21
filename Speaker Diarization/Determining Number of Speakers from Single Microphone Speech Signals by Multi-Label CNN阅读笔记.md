# Determining Number of Speakers from Single Microphone Speech Signals by Multi-Label CNN

# Abstract & Conclusion

Specs of windowed noisy speech signals for 1talker, 2 talkers and 3+ talkers are used as inputs to a multi-label CNN.

A database of speech signals consisting of 1 talker, 2 talkers and 3+ talkers has been examined in the presence of noise at 5dB and 10dB SNR levels.

# DATASET & CNN INPUT CONSIDERED



# MULTI-LABEL CNN CLASSIFICATION

Performance is often improved when more than one loss function is considered. The architecture involving more than one loss function is named multi-label CNN, also named multi-task CNN.

To take into consideration differences between different 3+talkers situations, another CNN using the sigmoid loss function is used. The response of this sigmoid CNN is a value between 0.1 to 0.8 reflecting 1talker to 8talkers, respectively.

The above two CNNs with different loss functions are trained separately.

The 256 outputs of the fully connected layer of the softmax CNN and the 64 outputs of the fully connected layer of the sigmoid CNN are combined together feeding another fully connected neural network to form our multi-label CNN as illustrated in Fig. 4.

![image-20220122090712459](https://tva1.sinaimg.cn/large/008i3skNly1gym6dveczfj30lo1fmqad.jpg)

# EXPERIMENTAL RESULTS AND DISCUSSION

![image-20220122091206661](https://tva1.sinaimg.cn/large/008i3skNly1gym6ixlxxzj30u013cadh.jpg)

From the confusion matrices, it can be observed that the overlap between 2talkers and 3+talkers classes is the largest due to the similarities of their spectrograms.
