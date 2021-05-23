# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aJ0Kgl8Tq4qA71ZKwxxdTxBok9GLYrQ2
"""

# Commented out IPython magic to ensure Python compatibility.
#!git clone https://github.com/JordanFoss/STAT3007_Project.git
# %cd STAT3007_Project/



import numpy as np
import librosa
from librosa.effects import split
import matplotlib.pyplot as plt
import glob
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from sklearn.model_selection import train_test_split

from pre_process import *

class DatasetWrapper(Dataset):
      def __init__(self, X, y):
          self.X, self.y = X, y
    
      def __len__(self):
          return len(self.X)
    
      def __getitem__(self, idx):
          return self.X[idx], self.y[idx]
      
      def change_type(self, dtype):
    
          return DatasetWrapper(self.X.type(dtype),self.y.type(dtype))
      
      def dataset(self):
          return DatasetWrapper(self.X,self.y)
      
      def get_data(self):
          return self.X, self.y
      
def load_samples(model_folder,sampling_rate = 16000,
                 padding = True, 
                 truncating = True, 
                 normal = True,
                 statement_type = [1,2] ,duration = 2,
                 Noisfy = True):
    '''
    Every sample in the dataset and pre-process it. 
    NOTE: this is only applicable for clean dataset at the moment

    Parameters
    ----------
    model_folder : str
        path to the project folder. e.g. .../STAT3007_Project
    sr : int, optional
        sampling_rate. The default is 16000.
    padding : boolean, optional
        Whether to apply padding. The default is True.
    truncating : boolean, optional
        Wether to apply truncation. The default is True.
    normal : boolean, optional
        Wether to apply amplitude normalisation. The default is True.
    statement_type : list, optional
        statement type to include. 1 - dog, 2 - kids. The default is [1,2].
    duration : int, optional
        maximum duration of the audio to include. The default is 2.

    Returns
    -------
    X : list
        complete set of samples in a list. Each element is an array with the shape (freq,duration)
    y : list
        complete set of targets corresponding to the samples. Its values maps the 5 emotions to 0-4

    '''
    
    # load samples
    X = []
    y = []
    
    for folder_name in glob.glob(model_folder + '/Audio_Speech_Actors_01-24/*'):
        for actor_folder in glob.glob(folder_name + '/*'):
            for sample_path in glob.glob(actor_folder + '/*.wav'):
              
              sample_name = sample_path.split('/')[-1]
              
              emotion, intensity, repetition, statement, actor = tuple(sample_name.split('-')[:5])
        
              
              # skip unwanted emotions and normal intensity
              if emotion not in target_map:
                  continue
            
              # skip unwanted statements
              if statement == '01' and 1 not in statement_type:
                  continue
              
              if statement == '02' and 2 not in statement_type:
                  continue
        
              sample, sampling_rate = librosa.load(sample_path, sr = sampling_rate)
              
              # truncate the silence of the audio sample
              truncated_sample = sample
        
              if truncating:
                  truncated_sample = truncate_silence(sample)
              
              
              # check the difference between the maximum duration and the sample duration
              total_duration = truncated_sample.shape[0]
              diff_duration = total_duration - (duration * sampling_rate)
        
              #normalisation
              if normal:
                  truncated_sample = amp_normalisation(truncated_sample)
              
        
              #pre pading the sample with zeros if it is shorter than the maximum duration
              # else, include only the first 2 seconds of it
              padded_sample = truncated_sample
        
              if padding and diff_duration < 0:
                  padded_sample = pre_pad(truncated_sample, int(duration * sampling_rate))
                
              if Noisfy:
                  thing = np.random.random_sample()
                  if thing < 0.2:
                      color = NoiseColour.White
                  elif thing < 0.4:
                        color = NoiseColour.Brown
                  elif thing < 0.6:
                        color = NoiseColour.Violet
                  elif thing < 0.8:
                        color = NoiseColour.Blue
                  else:
                        color = NoiseColour.Pink
                  padded_sample = nosify(padded_sample, colour = color)
                  print("Here")
              spectrogram = mel_spectral_decomposition(padded_sample[:int(sampling_rate * duration)], sampling_rate)
        
              target = target_generation(sample_name)
        
              X.append(spectrogram)
              y.append(target)
    
    return X, y

def load_noisy_samples(model_folder):
    # load samples
    X = []
    y = []
    
    for gender_folder in glob.glob(model_folder + '/*'):
        for actor_folder in glob.glob(gender_folder + '/*'):
            for sample_path in glob.glob(actor_folder + '/*.wav'):
              
              sample_name = sample_path.split('/')[-1]
              
              emotion, intensity, repetition, statement, actor = tuple(sample_name.split('-')[:5])

              sample, sampling_rate = librosa.load(sample_path, sr = 16000)
              mel_spectrogram = data_gen(sample, sampling_rate ,2)
              
              target = target_generation(sample_name)
        
              X.append(mel_spectrogram)
              y.append(target)
    
    return X, y

def load_sets(X,y,train_ratio = [0.7,0.7], seed = [10,11]):
    '''
    Loading training and test sets. It has the option to split up to twice

    Parameters
    ----------
    X : list
        list of mel-spectrograms
    y : list
        emotion labels
    train_ratio : list, optional
        train_ratios of first and second split. The default is [0.7,0.7].
    seed : list, optional
        random seeds for first and second split. The default is [10,11].

    Returns
    -------
    data_sets : list
        list of train_test split subsets

    '''
    data_sets = []
    for i in range(len(train_ratio)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_ratio[i], random_state = seed[i])
        
        data_train = DatasetWrapper(X_train, y_train)
        data_test = DatasetWrapper(X_test, y_test)
        
        data_sets.append((data_train,data_test))
        X,y = X_test,y_test
        
    return data_sets
        
        
        

