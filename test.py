#import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from math import e, pi, log, factorial
from scipy.stats import poisson

#import pytorch libraries
import torch
from torch import Tensor
#import the data and functions we prepared
import wcAnalysis as f

#rename variables
train_input = f.train_input
Teams_target = f.Teams_target
Teams_main = f.Teams_main
Teams = f.Teams

#check the length of the number of teams and the training set
print(len(set(train_input.localTeam)))
print(len(train_input))

#The Laplace approximation
theta, record_loss1 = f.SGD_batch(train_input, f.log_MAP, f.gradient_log_MAP, theta = torch.zeros(63,1), nb_epochs = 200)

#visualise the results
fig = plt.figure()

plt.subplot(221)
plt.plot(record_loss1)
plt.title('Loss Epochs 0-200')
plt.ylabel('Loss')
plt.xlabel('Number of Epochs')

plt.subplot(222)
plt.plot(record_loss1[50:])
plt.title('Loss Epochs 50-200')
plt.ylabel('Loss')
plt.xlabel('Number of Epochs')

plt.subplot(223)
plt.plot(record_loss1[100:])
plt.title('Loss Epochs 100-200')
plt.ylabel('Loss')
plt.xlabel('Number of Epochs')

plt.subplot(224)
plt.plot(record_loss1[125:])
plt.title('Loss Epochs 125-200')
plt.ylabel('Loss')
plt.xlabel('Number of Epochs')
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.01, right=0.95, hspace=1,
                    wspace=0.35)
plt.savefig('Laplace_loss.png', dpi=fig.dpi, bbox_inches= 'tight', pad_inches=0.1)
plt.show()

#calculat the covariance matrix
hessian = f.Hess_log_MAP(theta, train_input)
covariance= -hessian.inverse()
mu = theta

#save results
np.savetxt("theta_laplace.csv", mu.numpy(), delimiter=",")
np.savetxt("theta_laplace.csv", covariance.numpy(), delimiter=",")
#display results
#print(mu)

#prediction of scores
scores = f.Laplace_sample(mu, covariance, 2000, 1000)
print(scores)

#save the scores in a manageable matrix
scores = scores.view(32,32,2)
scores_mat = scores[0]
for i in range(1, 32):
    scores_mat = torch.cat((scores_mat, scores.view(32,32,2)[i]), 1)

#example
print(f.results("England","Sweden", scores_mat))
print(f.results("Sweden", "England", scores_mat))

#save the scores as csv
np.savetxt("scores_laplace.csv", scores_mat.numpy(), delimiter=",")

#reloading
