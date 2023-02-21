# Deep Neural Mel-Subband Beamformer for In-car Speech Separation 阅读笔记

## [ICASSP 2023]
![](https://tva1.sinaimg.cn/large/008vxvgGly1h90u6bwausj31u00bc0us.jpg)

![](https://tva1.sinaimg.cn/large/008vxvgGly1h90uebgzjsj31uo0iwn58.jpg)
**Narrow-band (NB) processing:** Processing each freq bin individually. The network is shared across all frequencies since it learns a unique function across all freq bins.

**Full-band (FB) processing:** As FB-based systems estimate full-band spectrum in a single processing step.

**Subband (SB) processing:** An SB-system is widely used to address the trade-off between performance and computational cost by helping to minimize the overall model size. This is achieved by splitting a full-band spectrum into several contiguous frequency bands, which are independently processed like an NB-system.

**Mel-Subband processing:** splits frequencies in mel-scale rather than uniformly, combining fewer freq bins within a band at low freq compared to high freq.

[For narrow-band references, refer to https://quancs.github.io/]


