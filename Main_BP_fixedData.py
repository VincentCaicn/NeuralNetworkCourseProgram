# -*- coding: utf-8 -*-
"""
@author: Vincent Cai
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

### Main Code for BP NN ###

class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        x = F.tanh(self.hidden(x))
        x = self.predict(x)
        return x
    
if __name__ == '__main__':
    
    file_name = 'DataSet.mat'
    raw_data = scio.loadmat(file_name)
    x_train = raw_data['x_train']
    y_train = raw_data['y_train']
    x_test = raw_data['x_test']
    y_test = raw_data['y_test']
    
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    x_train = Variable(x_train.reshape(-1,1))
    y_train = Variable(y_train.reshape(-1,1))
    x_test = x_test.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    net = Net(n_input=1, n_hidden=7, n_output=1)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),lr=0.1)
    
    print('------ Start Training ------')
    running_loss = []
    
    for t in range(1000):

        optimizer.zero_grad()
        
        prediction = net(x_train)
        loss = criterion(prediction, y_train)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.item())
        
        if t % 99 == 0:
            plt.cla()
            plt.scatter(x_train.data.numpy(),y_train.data.numpy()) #绘制真实曲线
            plt.plot(x_train.data.numpy(),prediction.data.numpy(),'+r',lw=5)
            plt.text(0.5,0,'Loss='+str(loss.item()),fontdict={'size':20,'color':'red'})
            plt.pause(0.1)
            
    plt.ioff()
    plt.show()
    
    print('------      预测和可视化      ------')
    
    y_prediction = net(x_test)
    loss_prediction = criterion(y_prediction, y_test)
    loss_prediction = loss_prediction.item()
    
    plt.plot(x_test.data.numpy(),y_test.data.numpy(),'+b')
    plt.plot(x_test.data.numpy(),y_prediction.data.numpy(),'or')
    plt.show()
    
            






