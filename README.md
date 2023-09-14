# EmoMix
Code for the creating emotional speech mixtures on the RAVDESS dataset.

These instructions assume you are familiar with the Speechbrain framework https://github.com/speechbrain/speechbrain 

1. Download the RAVDESS dataset following the instructions on https://github.com/BUTSpeechFIT/RAVDESS2Mix
2. Instead of the RAVDESS2Mix Prep, run Emo2Mix_prep.py
3. To use the dataset within the Speechbrain framework, use the DataIO method provided in Emo2Mix_dataio.py
4. Speechbrain training scripts for ogBSS(follwing the RAVDESS2Mix mixing method) and TrueBSS (follwing our Emo2Mix mixing method) are provided, along with the yaml files provided in the hparams folder. Copy these files into the LibriMix recipe folder within Speechbrain.
