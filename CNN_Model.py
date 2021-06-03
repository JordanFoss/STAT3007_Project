import torch.nn as nn

class ConvNet_RGB(nn.Module):
    def __init__(self, contain_linear = False, filter_num = 8, kernel_size = (2,3), channels = 3):
        super(ConvNet_RGB, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(channels, filter_num, kernel_size = (2,3))
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
    
    
class ConvNet_MultiChannel(nn.Module):
    def __init__(self, contain_linear = False, filter_num = 8, kernel_size = (2,3), channels = 8):
        super(ConvNet_MultiChannel, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(channels, filter_num, kernel_size = (2,3))
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
              nn.Linear(24*6*2, 1024),
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


class ConvNet(nn.Module):
    def __init__(self, contain_linear = False, filter_num = 8, kernel_size = (2,3)):
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