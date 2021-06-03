import torch.nn as nn
import torch
import math 

def output_size(input_shape, kernel_size,padding = 0, dilation = 1, stride = 1):
    output_shape = (input_shape + (2 * padding) - dilation * (kernel_size - 1)-1)/stride
    
    output_shape = math.floor(output_shape + 1)
    
    return output_shape
    
def CNN_output_shape(input_dimension = (128,63),
                     kernel = (2,3),
                     layers = 2,
                     max_pooling = True,
                     max_kernel = (1,2)):
    
    row, column = input_dimension
    for i in range(layers):
        row = output_size(row, kernel[0])
        column = output_size(column, kernel[1])
        
        row = output_size(row,max_kernel[0],stride = 2)
        column = output_size(column, max_kernel[1], stride = 2)
        
    return row, column
   
class ConvNet(nn.Module):
    def __init__(self, contain_linear = False, filter_num = 8, kernel_size = (2,3), input_shape = (1,128,63)):
        super(ConvNet, self).__init__()
        
        channels,freq,times = input_shape
        
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(channels, filter_num, kernel_size = (2,3))
        self.conv2 = nn.Sequential(nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride = 2),
            nn.Dropout(0.25),
            nn.Conv2d(filter_num, 24, kernel_size = (2,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride = 2)
        )

        self.contain_linar = contain_linear

        if contain_linear:
            
          row,column = CNN_output_shape(input_dimension=(freq,times))
          self.linear = nn.Sequential(
              nn.Linear(24*row*column, 1024),
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
  
# RNN combined with CNN
class LRCN(nn.Module):
    def __init__(self, CNN, shape = (24,32,3)):
        super(LRCN, self).__init__()

        self.cnn = CNN
        self.shape = shape

        channel, freq, times = shape
        self.lstm_layers = nn.LSTM(freq*channel*times,256,num_layers = 2, bidirectional = True)
        self.linear = nn.Sequential(nn.Linear(256*2, 5))
        self.flatten = nn.Flatten()

    def forward(self, x, step_size = 21, use_cuda = False):

      if use_cuda:
        h_t = torch.zeros(4,x.shape[0] ,256, dtype=torch.float).to(x.device)
        c_t = torch.zeros(4,x.shape[0], 256, dtype=torch.float).to(x.device)

      else:
        h_t = torch.zeros(4,x.shape[0], 256, dtype=torch.float)
        c_t = torch.zeros(4,x.shape[0], 256, dtype=torch.float)
      
      look_ahead_time = 21
      for current_time in range(0,x.shape[-1], step_size):

        x_t = x[:,:,:,current_time:current_time+look_ahead_time]
        conv_x = self.cnn(x_t)

        conv_x_flat =  self.flatten(conv_x)

        conv_x_flat = conv_x_flat.reshape(1,conv_x_flat.shape[0],conv_x_flat.shape[1])

        output, (h_t, c_t) = self.lstm_layers(conv_x_flat, (h_t, c_t))

      decision_vec = self.linear(output[0])
      return decision_vec
class Encoder(nn.Module):
  def __init__(self, filters = 16, kernal_size = (4,2)):
    super(Encoder, self).__init__()
    (K, S) = (5, 1)
    self.conv = nn.Sequential(nn.Conv2d(1, filters, kernel_size= kernal_size, stride = 1),
                              nn.ReLU(),
                              nn.MaxPool2d(kernel_size = 2),
                              nn.ReLU(),
                              nn.Conv2d(filters,filters,kernel_size= kernal_size, stride = 1),
                              nn.ReLU(),
                              nn.MaxPool2d(kernel_size = 2),
                              nn.Conv2d(filters,filters,kernel_size= kernal_size, stride = 1),
                              nn.ReLU()
                              )
    
  def forward(self, x):
    x = self.conv(x)
    return x



class Decoder(nn.Module):
  def __init__(self, filters = 16, kernal_size = (4,2), upsample_size1=(61,27), upsample_size2=(126,60)):
    super(Decoder, self).__init__()
    (K, S) = (2, 1)
    self.conv = nn.Sequential(nn.ConvTranspose2d(filters,filters, kernel_size = kernal_size),
                              nn.ReLU(),
                              nn.Upsample(size = upsample_size1),
                              nn.ConvTranspose2d(filters,filters, kernel_size = kernal_size),
                              nn.ReLU(),
                              nn.Upsample(size = upsample_size2),
                              nn.ConvTranspose2d(filters,1, kernel_size = kernal_size),
                              )
    
  def forward(self, x):
    x = self.conv(x)
    return x

class Autoencoder(nn.Module):
  def __init__(self, filters = 16, kernal_size=(4,2), upsample_size1=(61,27), upsample_size2=(126,60)):
    super(Autoencoder,self).__init__()
    self.filters = filters
    self.kernal_size = kernal_size
    self.upsample_size1 = upsample_size1
    self.upsample_size2 = upsample_size2
    self.encoder = Encoder(filters, kernal_size)
    self.decoder = Decoder(filters, kernal_size, upsample_size1, upsample_size2)
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
