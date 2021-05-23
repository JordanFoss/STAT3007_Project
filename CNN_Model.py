import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from pre_process import *


class ConvNet(nn.Module):
    def __init__(self, contain_linear = False, filter_num = 14, kernel_size = (2,3)):
        super(ConvNet, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, filter_num, kernel_size = (2,3))
        self.conv2 = nn.Sequential(nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride = 2),
            nn.Dropout(0.25),
            nn.Conv2d(filter_num, 24, kernel_size = (2,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride = 2),
        )

        self.contain_linar = contain_linear

        if contain_linear:
          self.linear = nn.Sequential(
              nn.Linear(24*32*14, 1024),
              nn.Linear(1024, 5),
          )

    def forward(self, x, inspect_feature = False):

      first_layer = self.conv1(x)
      conv_x = self.conv2(first_layer)

      output_x = conv_x
      if self.contain_linar:
        conv_x_flat  = self.flatten(conv_x)
        output_x = self.linear(conv_x_flat)
      
      if inspect_feature:
        return first_layer,conv_x,output_x
      return output_x

def classification(prediction):
  classified = torch.argmax(prediction, dim = 1)
  return classified

def accuracy(y_pred, y_test):
  class_pred = classification(y_pred)

  accuracy = class_pred == y_test

  accuracy_percent = torch.count_nonzero(accuracy)/accuracy.shape[0]
  return accuracy_percent.item()

def train_model(data_train, data_test, net, loss, nepoch ,lr = 0.01, batch_size = -1, use_cuda = False, print_output = True):

  # appropriate data type for CPU or GPU
  device = None
  if use_cuda and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    net = net.to(device)
  else:
    dtype = torch.FloatTensor

  optimizer = optim.SGD(net.parameters(), lr = lr)
  data_train = data_train.dataset.change_type(dtype)
  data_test = data_test.dataset.change_type(dtype)

  data_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)

  for epoch in range(nepoch):
    for X_batch, y_batch in data_loader:
      y_batch = y_batch.type(torch.LongTensor)
      if use_cuda and device != None:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_batch = y_batch.type(torch.cuda.LongTensor)

      optimizer.zero_grad()

      # since all our values are negative, we convert them to positive


      pred = net(X_batch)
      Rn = loss(pred, y_batch)
      Rn.backward()
      optimizer.step()

    if print_output:
      print('epoch:', epoch)
      print('loss:',Rn.item())
      print('------------')

  print('final loss:', Rn.item())

  return net