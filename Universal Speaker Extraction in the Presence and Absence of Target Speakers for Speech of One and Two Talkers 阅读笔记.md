# Universal Speaker Extraction in the Presence and Absence of Target Speakers for Speech of One and Two Talkers 阅读笔记

# Abstract
Traditional speaker extraction models fail in scenarios when the target speaker is absent from the mixture.

Propose to handle speech mixtures with one or two talkers in which the target speaker can either be present or absent.

SE uses a spk's ref signal to extract the target spk's voice in a multi-talker speech mixture w/o any prior knowledge about the number of speakers.

In the presence of the target speaker, the model extracts the target speaker’s voice, and in the absence of the target speaker, the model is expected to output silence. We intro a joint training scheme with one unified loss func for all four conditions.

# Universal Speaker Extraction Conditions
