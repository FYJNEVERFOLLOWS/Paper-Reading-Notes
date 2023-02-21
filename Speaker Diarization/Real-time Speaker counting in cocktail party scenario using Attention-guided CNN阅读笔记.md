# Real-time Speaker counting in cocktail party scenario using Attention-guided CNN

# Abstract & Conclusion

Most current speech technology systems assume that the number of co-current speakers is known.

Propose a real-time, single-channel attention-guided CNN to estimate the number of active speakers in overlapping speech.

Using a CNN model to extract higher-level info from the speech spectral content, the attention mechanism summarizes the extracted info into a compact feature vector without losing critical info. Finally, the active speakers are classified using a FC network.



# Introduction

The majority of the systems assume that the number of concurrent sources is known in advance, which is not a realistic assumption in most applications. Therefore, a real-time, simple, yet effective speaker count estimation is needed to bridge the gap between research and real-world applications.



# Problem setup



# System design

Since one of the output classes is non-speech, the proposed system can also be considered as a combined real-time Speech Activity Detector (SAD) that detects non-speech segments as short as 0.2 secs.



The choice of 8 2D convolutional layers with 128 output channels, kernel size of 5*5, and 2 FC layers with 256 neurons each is optimum.



Weighted Accuracy is Accuracy calculated within each class.



Crowd++ used Error Count Distance (ECD) to evaluate their method.