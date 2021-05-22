# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:04:47 2021

@author: qq871
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import scipy
import numpy as np
import librosa
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import pandas as pd
import glob
from librosa import display

from IPython.display import Audio

def train_model(data, net, loss, nepoch ,lr = 0.01, batch_size = -1, use_cuda = False, optimiser_type = 'SGD',print_output = True, classification = False, test_data = None):
  '''
    

    Parameters
    ----------
    data : Dataset
        training dataset
    net : nn.Module
        DL model
    loss : function
        loss function handle
    nepoch : int
        number of epoches
    lr : float, optional
        learning rate. The default is 0.01.
    batch_size : int, optional
        nunmber of samples in one batch. The default is -1.
    use_cuda : boolean, optional
        whether to use cuda. The default is False.
    optimiser_type : str
        type of optimiser to use
    print_output : boolean, optional
        whether to print output. The default is True.
    classification : boolean, optional
        is classification involved? This changes the type of target to longTensor. The default is False.
    test_data : Dataset, optional
        test dataset for loss over epoch. The default is None.

    Returns
    -------
    net : nn.Module
        trained network
    
    train_losses: list
        average batch losses over epoch
    
    test_losses: list
        test losses over epoch. None if test_data == None

    '''
  
  # appropriate data type for CPU or GPU
  device = None
  if use_cuda and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    net = net.to(device)
  else:
    dtype = torch.FloatTensor

  if optimiser_type == 'SGD':
      optimizer = optim.SGD(net.parameters(), lr = lr)
  else:
      
     optimizer = optim.Adam(net.parameters(), lr = lr)
  data = data.dataset.change_type(dtype)
  
  if test_data != None:
      X_test, y_test = test_data.dataset.get_data()
      y_test = y_test.type(torch.LongTensor)
      
      test_losses = []
  

  if batch_size == -1:
    data_loader = DataLoader(data,
                         batch_size = data.dataset__len__, shuffle = True)
  
  else:
    data_loader = DataLoader(data,
                             batch_size = batch_size, shuffle = True)
    
  train_losses = []
  for epoch in range(nepoch):
     batch_losses = []
     for X_batch, y_batch in data_loader:
      if use_cuda and device != None:
        X_batch = X_batch.to(device)
        
        if classification:
          y_batch = y_batch.type(torch.cuda.LongTensor)
        y_batch = y_batch.to(device)
      else:
          if classification:
            y_batch = y_batch.type(torch.LongTensor)

      optimizer.zero_grad()

      pred = net(X_batch)
      Rn = loss(pred, y_batch)
      Rn.backward()
      optimizer.step()
      batch_losses.append(Rn.item())
     avg_loss = np.mean(np.array(batch_losses))
     
     train_losses.append(avg_loss)
     if test_data != None:
         net_test = net.to(torch.device('cpu'))
         
         test_pred = net_test(X_test)
         test_error = loss(test_pred, y_test)
         test_losses.append(test_error.item())
         
     
     if print_output:
      print('epoch:', epoch)
      print('loss:', avg_loss)
      print('------------')
  
  print('final loss:', avg_loss)
  
  if test_data != None:
    return net, train_losses, test_losses
  else:
    return net, train_losses, None
