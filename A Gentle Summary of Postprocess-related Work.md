# GAN-related
## [Interspeech 2017] [Generative Adversarial Network-Based Postfilter for STFT Spectrograms](https://www.isca-speech.org/archive_v0/Interspeech_2017/pdfs/0962.PDF)

Generated spectra typically lack the fine structures that are close to those of the true data. Propose a GAN-based postfilter that is implicitly optimized to match the true feature distribution in adversarial learning.

GAN cannot be easily trained for very high-dimensional data such as STFT spectra. Thus take divide-and-concatenate strategy: first divide the spectrograms into multiple freq bands with overlap, reconstruct the individual bands using the GAN-based postfilter trained for each band, and connect th bands with overlap.

![](https://raw.githubusercontent.com/FYJNEVERFOLLOWS/Picture-Bed/main/202205/20220613095620.png)

## [Wavecyclegan2: Time-domain neural post-filter for speech waveform generation](https://arxiv.org/abs/1904.02892)