# [多通道语音分离] Localization Based Sequential Grouping for Continuous Speech Separation 阅读笔记

# Abstract
This study investigates robust speaker localization for continuous speech separation and speaker diarization, where we use speaker directions to group non-contiguous segments of the same speaker. Assuming that speakers do not move and are located in different directions, the DOA info provides an informative cue for accurate sequential grouping and speaker diarization.

Our system is block-online: given a block of frames with at most two speakers, we apply a two-speaker separation model to separate (and enhance) the speakers, estimate the DOA of each separated speaker, and group the separation results across blocks based on the DOA estimates.

# Introduction
