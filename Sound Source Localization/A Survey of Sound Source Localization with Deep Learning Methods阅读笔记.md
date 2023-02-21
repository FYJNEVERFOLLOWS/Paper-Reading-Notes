https://arxiv.org/abs/2109.03465



Sound Source Localization (SSL) is the technology of estimating the position of one or several sound sources from the multichannel signals captured by the microphone array. In most practical cases, SSL is simplified to the estimation of the sources' Direction-of-Arrival (DOA), i.e. it focuses on the estimation of azimuth and elevation angles, without estimating the distance to the microphone array.



SSL's applications: source separation, speech recognition, speech enhancement or human-robot interaction.



Even if the relationship between the information contained in the multichannel signal and the source(s) location is generally complex (especially in a multisource reverberant and noisy configuration), DNNs are powerful models that are able to automatically identify this relationship and exploit it for SSL, given that they are provided with a sufficiently large and representative amount of training examples.

![image-20211103094319813](https://tva1.sinaimg.cn/large/008i3skNly1gw1purj9hoj30m804x3yt.jpg)

A recent trend is to skip the feature extraction module to directly feed the network with multichannel raw data.



The major drawback of the DNN-based approaches is the lack of generality.



# ACOUSTIC ENVIRONMENT AND SOUND SOURCE CONFIGURATIONS

C. Number of sources
detect the activity of the source at the same time as the localization algorithm. For example, an additional neuron has been added to the output layer of the DNN used in [52], which outputs 1 when no source is active (in that case all other localization neurons are trained to output 0), and 0 otherwise.



The specific case where we have several speakers taking speechturns with or without overlap is strongly connected to thespeaker diarization problem (“who speaks when?”) [53], [54],[55]. Speaker localization, diarization and (speech) source sep-aration are intrinsically connected problems, as the informationretrieved from solving each one of them can be useful foraddressing the others [23], [56], [57]. An investigation of thoseconnections is out of the scope of the present survey.

《Keyword-based speaker localization: localizing a target speaker in a multi-speaker environment》



# Neural network architectures for SSL

C. CNN
