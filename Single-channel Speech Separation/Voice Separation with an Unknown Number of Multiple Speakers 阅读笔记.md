# Voice Separation with an Unknown Number of Multiple Speakers 阅读笔记

# Abstract
This method employs gated neural networks that are trained to separate the voices at multiple processing steps, while maintaining the speaker in each output channel fixed.

# Intro
Focus on the problem of supervised voice separation from a single microphone, which has seen a great leap in performance following the advent of DNNs. The current leading methodology is based on an overcomplete set of linear filters, and on separating the filter outputs at every time step using a mask for two speakers, or a multiplexer for more speakers. Since the order of the speakers is considered arbitrary (it is hard to sort voices), one uses a permutation invariant loss during training such that the permutation that minimizes the loss is considered.

The mask needs to extract and suppress more from the representation as the number speakers increases. We set out to build a mask-free method.
