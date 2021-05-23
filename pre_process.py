# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pVPXIvI9BJrbCCP9iykV4OvDgq10K190
    
This file contains several helper funtion for pre-processing
"""

import numpy as np
import librosa
from librosa.effects import split
import matplotlib.pyplot as plt
import glob
import torch
import colorednoise as cn

# produce emotion label

# 0 - calm; 1 - happy; 2 - sad; 3 - angry; 4 - surprised
target_map = {'02':0,'03':1,'04':2,'05':3,'08':4}

#Full Target Map
#target_map = {'01':0,'02':1,'03':2,'04':3,'05':4,'06':5,'07':6,'08':7}

class NoiseColour:
  White = 0
  Violet = -2
  Blue = -1
  Pink = 1
  Brown = 2

def nosify(samples, noise_level = 1, colour = NoiseColour.White):
  if colour == NoiseColour.White:
    noise = torch.randn_like(samples)
  else:
    noise = torch.from_numpy(cn.powerlaw_psd_gaussian(exponent = colour, size = samples.shape[1])).float()

  scaled_noise = noise * torch.mean(torch.abs(samples)) * noise_level
  
  noisy_samples = samples + scaled_noise
  return noisy_samples

def find_min_max():
  min_time = 41241
  max_time = 0
  min_sample = 0
  max_sample = 0
  for folder_name in glob.glob('./Audio_Speech_Actors_01-24/*'):
    for actor_folder in glob.glob(folder_name + '/*'):
      for sample_path in glob.glob(actor_folder + '/*'):
        sample_name = sample_path.split('/')[-1]
        file_format = sample_name.split('.')[-1]

        # we want only audio files, selected emotions and strong intensity
        if file_format != 'wav' or sample_name[:2] not in target_map or sample_name[3:5] == '01' or sample_name[6:8] == '01':
          continue
        
        sample, sampling_rate = librosa.load(sample_path, sr = None)

        sample = truncate_silence(sample)

        sampling_time = sample.shape[0]/sampling_rate

        if sampling_time < min_time:
          min_time = sampling_time
          min_sample = sample.shape[0]
          min_file = sample_name

        if sampling_time > max_time:
          max_time = sampling_time
          max_sample = sample.shape[0]
          max_file = sample_name

  return min_time, max_time, min_sample, max_sample, min_file, max_file

def spec_normlisation(spectrogram):
  mean = torch.mean(spectrogram)
  std = torch.std(spectrogram)

  norm = (spectrogram - mean)/std
  return norm

def target_generation(file_name):
    '''
    Generates the target of that audio file. Target will be one of the 5 emotions
    
    Parameter:
        file_name:str
            file_path of audio file
    Returns:
        target: int
            index that represents the target
    '''
    labels = file_name.split('.')[0].split('-')
    emotion = labels[0]

    if emotion not in target_map:
        return None
    
    return target_map[emotion]

# mel_decomposition
def mel_spectral_decomposition(sample,sampling_rate, title = ' title placeholder', visualise = False):
    '''
    Computes the mel-scale log-spectrogram.        
    
    Parameters
    ----------
    sample : numpy.array
        audio samples
    sampling_rate : int
        sampling rate of the audio
    title : str, optional
        title of the spectrogram plot. Ignore if visualise is False. The default is ' title placeholder'.
    visualise : boolean, optional
        boolean value to generate plot. The default is False.

    Returns
    -------
    spectrogram : numpy.ndarray
        array representation of spectrogram.Shape: (freq,times)

    '''

    spectrogram = librosa.feature.melspectrogram(y=sample, sr=sampling_rate, n_mels=128,fmax=8000) 
    spectrogram = librosa.power_to_db(spectrogram)
    
    if visualise:
      librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time', sr =sampling_rate);
      plt.title(title)
      plt.colorbar(format='%+2.0f dB')
    
    return spectrogram

def truncate_silence(sample):
    '''
    Truncates silence timesteps before and after the speech 
    

    Parameters
    ----------
    sample : numpy.array
        audio samples

    Returns
    -------
    truncated_sample: numpy.array
        truncated sample

    '''
    speech = np.where(abs(sample) > 1e-3)
    
    start = speech[0][0]
    end = speech[0][-1]
    
    return sample[start:end+1]

def truncate_silence_ex(sample):
    '''
    Truncates silence timesteps before and after the speech 
    This one uses librosa function instead of our own function
    

    Parameters
    ----------
    sample : numpy.array
        audio samples

    Returns
    -------
    truncated_sample: numpy.array
        truncated sample

    '''
    speech = split(sample, 20)
    start = speech[0][0]
    end = speech[-1][-1]
    
    return sample[start:end+1]

def amp_normalisation(sample):
    '''
    Normalise the amplitude of audio file.
    i.e [x - mean(x)]/std(x)

    Parameters
    ----------
    sample : numpy.array
        audio sample

    Returns
    -------
    normalised_sample: numpy.array
        normalised_sample

    '''
    mean = np.mean(sample)
    std = np.std(sample)
    return (sample - mean)/std

def pre_pad(samples, max_sample):
    '''
    pad the samples to the max number of samples

    Parameters
    ----------
    samples : numpy.array
        audio samples
    max_sample : int
        max number of samples

    Returns
    -------
    padded_sample : numpy.array
        padded audio sample

    '''
    
    sample_duration = samples.shape[0]
    
    num_to_pad = max_sample - sample_duration
    padded_sample = np.pad(samples,(num_to_pad,0),'constant', constant_values = 0)
    
    return padded_sample

def data_gen(sample, sampling_rate ,duration):
    '''
    This function generates the data for deep learning uses.
    It pre-processes each raw audio sample by:
        1. truncate silence
        2. normalise waveform
        3. pre-pad or crop to duration
        4. compute  mel-spectrogram

    Parameters
    ----------
    sample : numpy.array
        audio samples
    sampling_rate : int
        sampling rate of audio
    duration : int
        duration to keep or pre-pad to

    Returns
    -------
    spectrogram : numpy.ndarray
        mel-spectrogram of samples

    '''
    
    truncated_sample = truncate_silence(sample)
    truncated_sample = amp_normalisation(truncated_sample)
    
    total_duration = truncated_sample.shape[0]
    diff_duration = total_duration - (duration * sampling_rate)
    
    padded_sample = truncated_sample
    if diff_duration < 0:
        padded_sample = pre_pad(truncated_sample, int(duration * sampling_rate))
      
    spectrogram = mel_spectral_decomposition(padded_sample[:int(sampling_rate * duration)], sampling_rate)

    return spectrogram

def jordan_gen(sample, sampling_rate ,duration):
    '''
    This function generates the data for deep learning uses.
    It pre-processes each raw audio sample by:
        1. truncate silence
        2. normalise waveform
        3. pre-pad or crop to duration
        4. compute  mel-spectrogram

    Parameters
    ----------
    sample : numpy.array
        audio samples
    sampling_rate : int
        sampling rate of audio
    duration : int
        duration to keep or pre-pad to

    Returns
    -------
    spectrogram : numpy.ndarray
        mel-spectrogram of samples

    '''
    
    truncated_sample = truncate_silence(sample)
    truncated_normal = amp_normalisation(truncated_sample)
    
    total_duration = truncated_normal.shape[0]
    diff_duration = total_duration - (duration * sampling_rate)
    
    padded_sample = truncated_normal
    if diff_duration < 0:
        padded_sample = pre_pad(truncated_normal, int(duration * sampling_rate))
    else:
        padded_sample = padded_sample[:int(sampling_rate * duration)]
      
    spectrogram = mel_spectral_decomposition(padded_sample[:int(sampling_rate * duration)], sampling_rate)

    return truncated_sample,truncated_normal, padded_sample,spectrogram
