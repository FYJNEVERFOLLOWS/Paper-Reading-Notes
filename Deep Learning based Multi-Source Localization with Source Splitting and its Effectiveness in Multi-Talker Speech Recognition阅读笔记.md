# Deep Learning based Multi-Source Localization with Source Splitting and its Effectiveness in Multi-Talker Speech Recognition阅读笔记



# Abstract & Conclusion

Proposes a novel supervised learning method using deep neural networks to estimate the DOA of all the speakers simultaneously from the audio mixture. At the heart of the proposal is a source splitting mechanism that creates source-specific intermediate representations inside the network.

Proposed a novel deep learning based model for multi-source localization that can classify DOAs with a high resolution.

Source splitting model was shown to have a lower prediction error compared to a multi-label classification model.

Proposed a soft earth mover distance (SEMD) loss function for the localization task that models inter-class relationship well for DOA estimation and hence predicts near perfect DOA.



# Introduction

The localization component aids in source separation and recognition.

The learning is made more robust by encapsulating the feature extraction inside the neural network.

The source splitting mechanism involves two components.

The first component disentangles the input mixture by implicitly splitting them as source-specific hidden representations to enable tackling of multi-source DOA estimation via single-source DOA estimation.

The second component then maps these features to as many DOA posteriors as the number of sources in the mixture.





Drawback: Incorporate the 
