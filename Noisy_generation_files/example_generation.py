import glob
import numpy as np
import soundfile as sf
import os
import argparse
import librosa
import configparser as CP
from audiolib import audioread, audiowrite, snr_mixer

# Male 11
actor_path = '/Volumes/YuziSSD1/STAT3007/project/STAT3007_Project/Audio_Speech_Actors_01-24/Male/Actor_11'
noise_dir = '/Volumes/YuziSSD1/STAT3007/project/MS-SNSD/noise_train'
output_path = '/Volumes/YuziSSD1/STAT3007/project/STAT3007_Project/sample-noisy-speech-actor-11'

clean_to_use = []

emotion_to_use = ['02', '03', '04', '05', '08']
intensity_to_use = '02'
statement_to_use = '02'
rep_to_use = '01'


for clean_file in glob.glob(os.path.join(actor_path, '*.wav')):
    basename = os.path.basename(clean_file)
    series = basename.split(".")[0].split("-")
    if series[0] in emotion_to_use:
        if series[1] == intensity_to_use:
            if series[2] == statement_to_use:
                if series[3] == rep_to_use:
                    print(basename)
                    clean_to_use.append(clean_file)

for f_clean in clean_to_use:
    clean_samples, clean_sampling_rate = librosa.load(f_clean, sr = None)
    for f_noise in glob.glob(os.path.join(noise_dir, '*.wav')):
        noisy_samples, noisy_sampling_rate = librosa.load(f_noise, sr = clean_sampling_rate)
        # cut the noise samples, will cut further
        noisy_samples = noisy_samples[:noisy_sampling_rate * ((len(clean_samples)//clean_sampling_rate) + 1)]
        if len(noisy_samples)>=len(clean_samples):
                        # print("Yes!")
            noisy_samples = noisy_samples[0:len(clean_samples)]
        else:
            while len(noisy_samples)<len(clean_samples):
                newnoise, fs = librosa.load(f_noise)
                noiseconcat = np.append(noise, np.zeros(int(fs*0)))
                noise = np.append(noiseconcat, newnoise)

        noisy_samples = noisy_samples[0:len(clean_samples)]
        clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean_samples, noise=noisy_samples, snr=30)
        noise_name = os.path.basename(f_noise).split(".")[0]
        clean_name = os.path.basename(f_clean).split(".")[0]
        output_filename = clean_name + '_' + noise_name + '.wav'
        output_filename = os.path.join(output_path, output_filename)
        audiowrite(noisy_snr, 48000, output_filename, norm=False)
