# A Real-Time Speaker Diarization System Based on Spatial Spectrum



# Abstract & Conclusion

Several challenges in speaker diarization: 

(1) to segment and separate overlapping speech from two speakers;

(2) to estimate the number of speakers when participants may enter or leave the conversation at any time;

(3) to provide accurate speaker identification on short text-independent utterances;

(4) to track down speakers movement during the conversation;

(5) to detect speaker change incidence real-time.



# 1. Introduction

Speaker diarization is a process of finding the optimal segmentation based on speaker identity and determining the identity of each segments. i.e. "who speaks when". Match a list of audio segments to a list of different speakers.

Propose a speaker diarization system that effectively incorporates spatial information.

Microphone array contributes to our speaker diarization system in two ways.

(1) the ability to localize sound source enables the system to find the optimal segmentation points with remarkable accuracy. Locations of each segments are effective complements for speaker embeddings in joint clustering, especially for short segments. 

(2) differential directional microphone array significantly improves the quality of speaker's voice in far-field, noisy environment, which in turn enhances the representative power of speaker embeddings.



# 2. System Description

Our speaker diarization system with spatial spectrum (SDSS).

![image-20220203211624526](https://tva1.sinaimg.cn/large/008i3skNly1gz0mw9c8mqj31y00u0103.jpg)

## 2.1. Audio segmentation based on beamforming

Audio segmentation and finding the exact point in time of a speaker change incidence are determined by the joint efforts of spatial localization and NN-VAD.

The output signals of the beamformers are spatially separated from each other.

Circular Differential Directional Microphone Array is based on a uniform circular array with directional microphones depicted in Fig.1. All the directional elements are uniformly distributed on a circle and directions are pointing outward.

![image-20220203212215898](https://tva1.sinaimg.cn/large/008i3skNly1gz0n2anhl3j30yg0pkmyz.jpg)

The output angle goes through an online clustering one after another. Every time an angle incidence that lies outside of the current cluster is spotted, we mark that current frame as a possible speaker change timestamp.

## 2.2. Speaker diarization

Consider two utterances $u_A$ and $u_B$. Traditional speaker verification task tries to estimate the probability that $u_A$ and $u_B$ are from the same speaker, $P(same|u_A,u_B)$. 

With additional estimated DOA $d_A$ and $d_B$, assuming that they are independent from $u_A$ and $u_B$, we instead try to estimate the joint conditional probability $P(same|u_A,u_B)P(same|d_A,d_B)$.



An online agglomerative hierarchical clustering (AHC) is performed on the audio segments and source location, based on the joint conditional probability.



Let $S = \{s_1,s_2,...,s_N\}$ be the list of current active speakers in the session and $D = \{d_1,d_2,...,d_N\}$ be the corresponding location of these speakers. Let $u'$ be a new incoming speech segment and $d'$ be the source localization of that speech segment. We define the following probability:
$$
\begin{aligned}
&p_{\text {new }}=P\left(u^{\prime} \notin s, \forall s \in S \mid S, D, u^{\prime}, d^{\prime}\right) \\
&p_{\text {update }}=P\left(u^{\prime} \in s, s \in S, \angle\left(d_{s}, d^{\prime}\right)<T_{d} \mid S, D, u^{\prime}, d^{\prime}\right) \\
&p_{\text {move }}=P\left(u^{\prime} \in s, s \in S, \angle\left(d_{s}, d^{\prime}\right) \geq T_{d} \mid S, D, u^{\prime}, d^{\prime}\right)
\end{aligned}
$$
When running real-time, we update the state of conversation $(S,D)$ every time we receive a new sample segment $(u',d')$.

## 2.4. Separation of overlapping speech

Beamforming allows us to separate signals from different DOAs.



Useful References
