import glob
import numpy as np
import soundfile as sf
import os
import argparse
import librosa
import configparser as CP
from audiolib import audioread, audiowrite, snr_mixer


# some parameters
SNR = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
silence_length = 0

# clean sample for actor 02 
clean_sample = '/Volumes/YuziSSD1/STAT3007/project/STAT3007_Project/Audio_Speech_Actors_01-24/Female/Actor_02/01-01-01-01-02.wav'
# noise_sample = '/Volumes/YuziSSD1/STAT3007/project/MS-SNSD/noise_train/Babble_1.wav'
noise_sample = '/Volumes/YuziSSD1/STAT3007/project/test/test.wav'


# output path
output_base_dir = '/Volumes/YuziSSD1/STAT3007/project/STAT3007_Project/noisy-test'


# clean, fs1 = audioread(clean_sample)
# noise, fs2 = audioread(noise_sample)
# print("The clean frequency is {}".format(fs1))
# print("The noise frequency is {}".format(fs2))

# if len(noise)>=len(clean):
#     print("Yes!")
#     noise = noise[0:len(clean)]
# else:
#     while len(noise)<len(clean):
#         newnoise, fs = audioread(noise_sample)
#         noiseconcat = np.append(noise, np.zeros(int(fs*silence_length)))
#         noise = np.append(noiseconcat, newnoise)

# noise = noise[0:len(clean)]

clean, fs1 = librosa.load(clean_sample, sr=None)
noise, fs2 = librosa.load(noise_sample, sr=None)
# print("The clean frequency is {}".format(fs1))
# print("The noise frequency is {}".format(fs2))

print(len(clean))
print(len(noise))

if len(noise)>=len(clean):
    print("Yes!")
    noise = noise[0:len(clean)]
else:
    while len(noise)<len(clean):
        newnoise, fs = librosa.load(noise_sample)
        noiseconcat = np.append(noise, np.zeros(int(fs*silence_length)))
        noise = np.append(noiseconcat, newnoise)

noise = noise[0:len(clean)]



for snr in SNR:
    print("=========== SNR {} ============".format(snr))
    clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean, noise=noise, snr=snr)
    output_filename = os.path.join(output_base_dir, 'SNR_{}.wav'.format(snr))
    # output_filename = 'SNR_{}'.format(snr)
    audiowrite(noisy_snr, 48000, output_filename, norm=False)

