import librosa
import glob
import os
from librosa import display
import soundfile as sf
import numpy as np


clean_dir = '/Volumes/YuziSSD1/STAT3007/project/STAT3007_Project/Audio_Speech_Actors_01-24'
noise_dir = '/Volumes/YuziSSD1/STAT3007/project/MS-SNSD/noise_train'


noise_length = []
for noise_file in glob.glob(os.path.join(noise_dir, '*.wav')):
    noisy_samples, noisy_sampling_rate = librosa.load(noise_file, sr = None)
    # print(noisy_sampling_rate)
    noise_length.append(len(noisy_samples))

noise_length_cut = []
for i in ['Female', 'Male']:
    path = os.path.join(clean_dir, i)
    for actor_dir in os.listdir(path):
        directory = os.path.join(path, actor_dir)
        for f in glob.glob(os.path.join(directory, '*.wav')):
            clean_samples, clean_sampling_rate = librosa.load(f, sr = None)
            noise_length_cut.append(16000 * ((len(clean_samples)//clean_sampling_rate) + 1))

print(np.min(noise_length))
print(np.max(noise_length_cut))