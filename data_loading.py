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
        
        
        

