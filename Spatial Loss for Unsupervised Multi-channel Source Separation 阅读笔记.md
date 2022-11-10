# Spatial Loss for Unsupervised Multi-channel Source Separation 阅读笔记

# Abstract
Propose a spatial loss for unsupervised multi-channel source separation. The proposed loss exploits the duality of direction of arrival (DOA) and beamforming: the steering and beamforming vectors should be aligned for the target source, but orthogonal for interfering ones. The spatial loss encourages consistency between the mixing and demixing systems from a classic DOA estimator and a neural separator. With the proposed loss, we train the neural separators based on minimum variance distortionless response (MVDR) beamforming and independent vector analysis (IVA). We also investigate the effectiveness of combining our spatial loss and a signal loss, which uses the outputs of blind source separation as the references. 

# 1. Intro
Propose a spatial loss func for unsupervised multi-channel source separation that enforces consistency of the estimated spatial parameters.