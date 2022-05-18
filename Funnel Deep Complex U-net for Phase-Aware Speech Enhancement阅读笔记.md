# Funnel Deep Complex U-net for Phase-Aware Speech Enhancement阅读笔记

# Abstract & Conclusion

Most of the early models focused on estimating the magnitude of spectrum while ignoring the phase.

The encoder-decoder structure in Deep Complex U-net (DCU) has been proven to be effective for complex-valued data.

Design the FDCU, which could process mag info and phase info separately through one-encoder-two-decoders structure.

Our model incorporated the masking and mapping based method to estimate clean waves from noisy.

Designed a new loss func S-SISNR, which can further improve the performance of the model at low SNR.