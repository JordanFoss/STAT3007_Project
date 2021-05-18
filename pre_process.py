# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pVPXIvI9BJrbCCP9iykV4OvDgq10K190
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import glob

# produce emotion label

# 0 - calm; 1 - happy; 2 - sad; 3 - angry; 4 - surprised
target_map = {'02':0,'03':1,'04':2,'05':3,'08':4}

def target_generation(file_name):
  labels = file_name.split('.')[0].split('-')
  emotion = labels[0]

  if emotion not in target_map:
    return None

  return target_map[emotion]



# mel_decomposition
def mel_spectral_decomposition(sample,sampling_rate, title = ' title placeholder', visualise = False):

  spectrogram = librosa.feature.melspectrogram(y=sample, sr=sampling_rate, n_mels=128,fmax=8000) 
  spectrogram = librosa.power_to_db(spectrogram)

  if visualise:
    librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time', sr =sampling_rate);
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')

  return spectrogram

def truncate_silence(sample):
  speech = np.where(abs(sample) > 1e-3)

  start = speech[0][0]
  end = speech[0][-1]

  return sample[start:end+1]

def amp_normalisation(sample):
  mean = np.mean(sample)
  std = np.sqrt(np.var(sample))
  return (sample - mean)/std

def pre_pad(samples, max_sample):
  sample_duration = samples.shape[0]

  num_to_pad = max_sample - sample_duration
  padded_sample = np.pad(samples,(num_to_pad,0),'constant', constant_values = 0)

  return padded_sample

def load_samples(padding = True, truncating = True, normal = True, duration = 2):
  # load samples
  X = []
  y = []

  for folder_name in glob.glob('./Audio_Speech_Actors_01-24/*'):
    for actor_folder in glob.glob(folder_name + '/*'):
      for sample_path in glob.glob(actor_folder + '/*'):
        
        sample_name = sample_path.split('/')[-1]
        file_format = sample_name.split('.')[-1]

        if file_format != 'wav' or sample_name[:2] not in target_map or sample_name[3:5] == '01' or sample_name[6:8] == '01':
          continue

        sample, sampling_rate = librosa.load(sample_path, sr = 16000)
        
        truncated_sample = sample

        if truncating:
          truncated_sample = truncate_silence(sample)
        

        total_duration = truncated_sample.shape[0]
        diff_duration = total_duration - (duration * sampling_rate)

        if normal:
          truncated_sample = amp_normalisation(truncated_sample)
        

        padded_sample = truncated_sample

        if padding and diff_duration < 0:
            padded_sample = pre_pad(truncated_sample, int(duration * sampling_rate))
          

        spectrogram = mel_spectral_decomposition(padded_sample[:int(sampling_rate * duration)], sampling_rate)

        target = target_generation(sample_name)

        if target != None:
          X.append(spectrogram)
          y.append(target)
  
  return X, y