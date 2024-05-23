# Acoustic-Scene-Classification

In this project, we have explored transformer(PaSST) and Convolution(Open-L3) based networks for acoustic Scene Classification. 

Problem Description
Variations in infrastructure, population density, cultural factors, and device resolution can bias audio features, leading to performance degradation in case of acoustic scenes as well as event classification tasks.
These kinds of variations in the available data introduce bias and leads to degradation in performance in deep classification models.
This bias necessitates the need a robust model design to address the above challenge of multiple domains which get introduced across different recording conditions.
This project  aims to develop a robust end to end Acoustic Scene Classification (ASC) system capable of adapting to these diverse recording conditions, regardless of cities or devices. 

Dataset Description

This Project utilises TAU Urban Acoustic Scenes 2020 Mobile, Development dataset provided by Dcase2020 challange for subtask-A (Acoustic Scene Classification).

Dataset Can be downloaded from https://zenodo.org/records/3819968

Dataset includes recordings from 10 European cities across 10 different acoustic scenes using 4 different devices. 
Devices Used:
Device A: Soundman OKM II Klassik/studio A3 electret binaural microphone + Zoom F8 audio recorder (main recording device)
Device B: Samsung Galaxy S7
Device C: iPhone SE
Device D: GoPro Hero5 Session

Synthetic data for 11 mobile devices (S1-S11) created based on recordings from device A and impulse responses from real devices.
Acoustic Scenes: The dataset covers ten acoustic scenes, including airports, indoor shopping malls, metro stations, pedestrian streets, public squares, streets with varying traffic levels, tram/bus/metro travel, and urban parks.

Dataset Size: Comprises 40 hours of data from device A, with smaller amounts from the other devices. Audio is provided in a single-channel 44.1kHz 24-bit format.

Simulated Data: Synthetic data was created using recordings from device A, impulse responses from real devices, and dynamic range compression to simulate realistic recordings.

For our results, we have omitted the extra devices not present in the train set from the test set in order to avoid open set classification.

