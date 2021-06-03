import glob
import numpy as np
import soundfile as sf
import os
import argparse
import librosa
import shutil

sample = '/Volumes/YuziSSD1/STAT3007/project/Noisy_Speech_Actors_01-24/Female/Actor_02/02-02-01-02-02_AirConditioner_1.wav'

all_noisy_samples_path = '/Volumes/YuziSSD1/STAT3007/project/Noisy_Speech_Actors_01-24_snr40'
upload_path = '/Volumes/YuziSSD1/STAT3007/project/SNR40-Noisy-samples-to-train-for-seminar'

def split_data(noisy_path, upload_path):
    for i in ['Female', 'Male']:
        path = os.path.join(noisy_path, i)
        output_dir_1 = os.path.join(upload_path, i)
        for actor_name in os.listdir(path):
            # print(actor_name)
            count = 0
            actor_dir = os.path.join(path, actor_name)
            output_dir_2 = os.path.join(output_dir_1, actor_name)
            if not os.path.exists(output_dir_2):
                os.mkdir(output_dir_2)
            for noisy_file in glob.glob(os.path.join(actor_dir, '*.wav')):
                basename = os.path.basename(noisy_file)
                output_dir_3 = os.path.join(output_dir_2, basename)
                if int(basename[-5]) < 5:
                    count += 1
                    shutil.move(noisy_file, output_dir_3)
            print(actor_name + ' ' + str(count) + ' files to move')

split_data(all_noisy_samples_path,upload_path)