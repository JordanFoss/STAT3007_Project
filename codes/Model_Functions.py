import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import numpy as np

def classification(prediction):
    '''
    

    Parameters
    ----------
    prediction : torch.tensor
        one-hot representation of prediction vector

    Returns
    -------
    classified : int
        numerical class label

    '''
    
    classified = torch.argmax(prediction, dim = 1)
    return classified

def accuracy(y_pred, y_test):
    '''
    

    Parameters
    ----------
    y_pred : torch.tensor
        batch prediction vector (one hot)
    y_test : torch.tensor
        batch true labels (numerical)

    Returns
    -------
    accuracy : float
        accuracy of predictions

    '''
    
    
    class_pred = classification(y_pred)
    
    accuracy = class_pred == y_test
    
    accuracy_percent = torch.count_nonzero(accuracy)/accuracy.shape[0]
    return accuracy_percent.item()

def train_model(data_train, 
                data_test, 
                net, loss, 
                nepoch , 
                lr = 0.01, 
                batch_size = -1, 
                momentum = 0,
                use_cuda = False, 
                print_output = True, 
                optimiser = 'SGD'):
    '''
    

    Parameters
    ----------
    data_train : DataWrapper
        training dataset
    data_test : DataWrapper
        testing dataset
    net : nn.Module
        instance of CNN model
    loss : nn.loss
        loss function handle
    nepoch : int
        number of epoches
    lr : float, optional
        learning rate. The default is 0.01.
    batch_size : int, optional
        batch size. The default is -1.
    momentum : float, optional
        momentum of optimiser. The default is 0.
    use_cuda : boolean, optional
        whether to use cuda. The default is False.
    print_output : boolean, optional
        prints output after every epoch training. The default is True.
    optimiser : str, optional
        choice of optimiser. The default is 'SGD'.

    Returns
    -------
    net : nn.Module
        trained model
    avg_loss : list
        average training loss over epoches
    avg_acc : list
        average training accuracy over epoches
    test_loss : list
        test loss over epoches
    test_acc : list
        test accuracy over epoches

    '''
    

    # setting up arrays for recording
    test_acc = []
    avg_acc = []
    
    test_loss = []
    avg_loss = []
    # appropriate data type for CPU or GPU
    device = None
    if use_cuda and torch.cuda.is_available():
      dtype = torch.cuda.FloatTensor
      device = torch.device("cuda")
      net = net.to(device)
    else:
      dtype = torch.FloatTensor
    
    if optimiser == 'SGD':
      optimizer = optim.SGD(net.parameters(), lr = lr, momentum = momentum)
    else:
      optimizer = optim.Adam(net.parameters(), lr = lr, momentum = momentum)
    data_train = data_train.change_type(dtype)
    data_test = data_test.change_type(dtype)
    
    X_test,y_test = data_test.get_data()
    
    y_test = y_test.type(torch.LongTensor)
    if device != None:
      y_test = y_test.type(torch.cuda.LongTensor)
    
    data_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
    
    for epoch in range(nepoch):
      batch_acc = []
      batch_loss = []
      for X_batch, y_batch in data_loader:
        
    
        y_batch = y_batch.type(torch.LongTensor)
        if use_cuda and device != None:
          X_batch = X_batch.to(device)
          y_batch = y_batch.to(device)
          y_batch = y_batch.type(torch.cuda.LongTensor)
    
          
    
        optimizer.zero_grad()
    
    
        pred = net(X_batch)
        Rn = loss(pred, y_batch)
        accur = accuracy(pred,y_batch)
    
        batch_acc.append(accur)
        batch_loss.append(Rn.to(torch.device('cpu')).detach().numpy())
    
        Rn.backward()
        optimizer.step()
    
    
      avg_batch_loss = np.mean(batch_loss)  
      avg_batch_acc = np.mean(batch_acc)
      avg_acc.append(avg_batch_acc)
      avg_loss.append(avg_batch_loss)
    
      pred = net(X_test)
      Rn = loss(pred, y_test)
      accur = accuracy(pred,y_test)
      test_acc.append(accur)
      test_loss.append(Rn.to(torch.device('cpu')).detach().numpy())
    
    
      if print_output:
        print('epoch:', epoch)
        print('loss:',Rn.item())
        print('------------')
      
    
    print('final loss:', Rn.item())
    
    return net, avg_loss, avg_acc, test_loss, test_acc

