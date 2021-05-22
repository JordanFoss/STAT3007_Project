import torch.nn as nn
from torch.utils.data import Dataset
import torch
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
              nn.Linear(1024, 8),
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

def classification(prediction):
  classified = torch.argmax(prediction, dim = 1)
  return classified

def accuracy(y_pred, y_test):
  class_pred = classification(y_pred)

  accuracy = class_pred == y_test

  accuracy_percent = torch.count_nonzero(accuracy)/accuracy.shape[0]
  return accuracy_percent.item()
