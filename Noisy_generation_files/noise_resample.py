import librosa
import os
from librosa import display
import soundfile as sf

# make the noise the same frequency and length as the clean speech 
def reduce_noise_length(noise_path, clean_sample_path, save_path):
    clean_samples, clean_sampling_rate = librosa.load(clean_sample_path, sr = None)
    noisy_samples, noisy_sampling_rate = librosa.load(noise_path, sr = clean_sampling_rate)
    
    print(len(clean_samples)//clean_sampling_rate)

    noisy_samples = noisy_samples[:noisy_sampling_rate * ((len(clean_samples)//clean_sampling_rate) + 1)]
    
    sf.write(save_path, noisy_samples, noisy_sampling_rate)
    
    return save_path


noise_dir = '/Volumes/YuziSSD1/STAT3007/project/MS-SNSD/noise_train'
noise_file_babble = '/Volumes/YuziSSD1/STAT3007/project/MS-SNSD/noise_train/Babble_1.wav'
clean_sample_path = '/Volumes/YuziSSD1/STAT3007/project/STAT3007_Project/Audio_Speech_Actors_01-24/Female/Actor_02/01-01-01-01-02.wav'
save_path = '/Volumes/YuziSSD1/STAT3007/project/test/test.wav'

reduce_noise_length(noise_file_babble, clean_sample_path, save_path)

