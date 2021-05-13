# STAT3007_Project
Teaching a robot to feel. 

Also the filename layout for the audio files is as follows
Filename example: 03-01-06-01-02-01-12.mp4 (audio-speech-emotion-intensity-statement-repetition-actor)

Audio-only (03) (Always 03 for us for obivous reasons)
<br>
Speech (01) (Always 01 for us unless we want to train on singing data)
<br>
Fearful (06) Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
<br>
Normal intensity (01) Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
<br>
Statement "dogs" (02) Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
<br>
1st Repetition (01 )Repetition (01 = 1st repetition, 02 = 2nd repetition).
<br>
12th Actor (12) Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


Jordan's refined filename layout:emotion-intensity-statement-repetition-actor.wav

## Example noisy data subset:
81 noises for the following:
<br>
5 emotions- strong intensity -dog statement - 1 rep - 1 actor
<br>
The 5 emotions: calm (02), happy(03), sad(04), angry(06), suprised(08)

## Architecure we are currently looking at:
1. discriminator: TBD
2. generator : CNN + RNN (https://ieeexplore.ieee.org/abstract/document/7820699 )

## Integrating Colab with Github
The following link shows all the available <code>.ipynb</code> files from our repo that can be opened by colab:
https://colab.research.google.com/github/JordanFoss/STAT3007_Project

More details can be found in <code>colab-github-demo.ipynb</code>
