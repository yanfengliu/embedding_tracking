# Simultaneous Amodal Instance Segmentation and Pixel-Level Multi-Object Tracking through Layered Deep Pixel Embeddings

This repo contains official source code that implements my Master's thesis work. 

The idea is to use pair of frames from a video as input and produce complete masks for every object in an image, including both its visible and invisible parts (amodal segmentation). Then track these objects by predicting their masks in the next frame. This is in contrast with the common approach towards MOT (Multi-Object Tracking) where objects are only described by bounding boxes. Multi-class MOT is also addressed in this work, in contrast with the more commonly researched single-class MOT. Common datasets usually only contain pedestrian annotations. This lack of data is a major factor that prohibits multi-class MOT. This project bypasses the contraints by using synthetic data. 
