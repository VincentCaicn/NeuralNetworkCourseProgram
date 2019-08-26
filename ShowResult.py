# -*- coding: utf-8 -*-
"""
@author: Vincent Cai
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.io as scio

file_name = 'Result.mat'
result_data = scio.loadmat(file_name)

x_test = result_data['x_test']
y_test = result_data['y_test']
y_prediction = result_data['y_prediction']

running_loss_MSE = result_data['running_loss_MSE'].reshape(-1,1)
running_loss_MAE = result_data['running_loss_MAE'].reshape(-1,1)
running_loss_R2_score = result_data['running_loss_R2_score'].reshape(-1,1)

x = np.linspace(1,1000, num=1000)
plt.plot(x, running_loss_MSE)
plt.show()