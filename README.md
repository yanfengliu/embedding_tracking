# Amodal Instance Segmentation and Multi-Object Tracking with Deep Pixel Embeddings

This repo contains official source code that implements my Master's thesis work. 

### Abstract

This thesis extends upon the representational output of semantic instance segmentation by explicitly including both visible and occluded parts. A fully convolutional network is trained to produce consistent pixel-level embedding across two layers such that, when clustered, the results convey the full spatial extent and depth ordering of each instance. Results demonstrate that the network can accurately estimate complete masks in the presence of occlusion and outperform leading top-down bounding-box approaches. 

The model is further extended to produce consistent pixel-level embeddings across two consecutive image frames from a video to simultaneously perform amodal instance segmentation and multi-object tracking. No post-processing motion modelling, identity matching, or Hungarian Algorithm is needed to perform multi-object tracking. The advantages and disadvantages of such a bounding-box-free approach are studied thoroughly. Experiments show that the proposed method outperforms the state-of-the-art bounding-box based approach on our simple yet challenging synthetic tracking dataset. 

### Key Illustrations

**Model architecture for simultaneous amodal instance segmentation and multi-class multi-object tracking**

![](https://i.imgur.com/aXiP8CE.png)

**Key Frame Identity linking**

![](https://i.imgur.com/5Nox1aR.png)

**Sample tracking results**

![](https://i.imgur.com/bXK0QO0.png)
