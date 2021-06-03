# STAT3007_Project
Teaching a robot to feel. 


# Instructions for runing training and testing files
Before running the notebooks with training_testing in the name, you will first have to create a shortcut to the google drive containing the noisy data. The link to this google drive is:
https://drive.google.com/drive/folders/1n9xwoN4oa4teVaBLyc5bvzuJZ70zhhQk?usp=sharing

# File names
Filename layout:emotion-intensity-statement-repetition-actor.wav

* Fearful (06) Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
* Normal intensity (01) Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
* Statement "dogs" (02) Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
* 1st Repetition (01 )Repetition (01 = 1st repetition, 02 = 2nd repetition).
* 12th Actor (12) Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Filename example: 06-01-02-01-12.mp4

# Pre-process steps:
1. load audio with downsampled sampling rate 16000Hz
2. truncate radio slience before and after each audio clip with a fixed, hand-crafted threshold
3. normalise amplitude waveform with 0 mean and unit variance
4. pick a certain initial duration of the audio sample (if shorter than the sampling duration, pre-pad the sample with zeros)
5. compute mel-spectrogram (amplitude -> power spectrum -> log-spectrogram -> mel-scaling)

# Train/Test split:
## CNN models:
1. split amongst authors (16/8)
2. induce noises

## Autoencoder Training:
1. Strafified sampling (see report)

# Example noisy data subset:
79 noises for the following:
<br>
5 emotions- strong intensity -dog statement - 1 rep - 1 actor
<br>
The 5 emotions: calm (02), happy(03), sad(04), angry(05), suprised(08)

# Architecture included in our report:
1. CNN
2. CNN + LSTM
3. CNN + RGB
4. RGB + CNN + LSTM
5. Autoencoder + CNN


# Integrating Colab with Github
The following link shows all the available <code>.ipynb</code> files from our repo that can be opened by colab:
https://colab.research.google.com/github/JordanFoss/STAT3007_Project

More details can be found in <code>colab-github-demo.ipynb</code>

# Presentation Slides
https://docs.google.com/presentation/d/1QSJ8ocBKJbPoVOcoXiKNEMI355Wn7ljikF_xhB7_KaM/edit#slide=id.p

