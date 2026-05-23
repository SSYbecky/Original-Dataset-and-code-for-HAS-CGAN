# Original-Dataset-and-code-for-HAS-CGAN
This repository includes dataset file, code file and results file.
The original dataset file consists of cutting force signals and machining parameters and real measured surface roughness.
The code consists of three parts which includes data preprocessing, data augmentation and prediction models.
The results file consists of some results generated from data preprocessing processes, saved trained CGAN models for inference and predictive results generated from prediction models. 
The following is about how to run this project to get the results of every step:
Data preprocessing:
compute_force & truncate valid data.py: This .py is to compute compound force and truncate the valid data from original force data for each sample.
Look at similarities.py: Use this .py to generate new force signals under different labels through well-trained generative models.
compute_WT_coherence.py: Use this .py to compute wavelet cohenrence of generated signals and original signals.

CGAN models:
HAS_CGAN.py: Train the HAs-CGAN model.

Predictive models:
SVR+LSTM+RF.py: This .py contains three different prediction models, which are SVR, LSTM and RF resepectively. 

