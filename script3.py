from __future__ import division
from sklearn import svm
import numpy as np
import pandas as pd
import scipy as scip
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
import torch
from torch.autograd import Variable
from utility import *
from sklearn import metrics

protvec = initialize()

# Positive and Negative Datasets
pos_data = []
neg_data = []

# Read in the Positive Dataset
with open('pos.data') as f:
    for line in f:
        line = line.rstrip()
        '''
        for AA in line:
            l.append(table_hot[AA])
        l.append(1.0)
        '''
        p = embedding(protvec,line)
        pos_data.append(p)

# Read in the Negative Dataset
with open('neg.data') as f:
    for line in f:
        n = []
        line = line.rstrip()
        '''
        for AA in line:
            l.append(table_hot[AA])
        l.append(0.0)
        '''
        n = embedding(protvec, line)
        neg_data.append(n)


# Data preparation
data = pos_data + neg_data
data = np.array(data)
np.random.shuffle(data)
data, test = np.array_split(data,2)
print len(data), len(test)

print data.tolist()

# 1D convolution
class Discrim(torch.nn.Module):
    def __init__(self):
        super(Discrim, self).__init__()
        self.c1 = torch.nn.Conv1d(1,20,3)
        self.relu = torch.nn.LeakyReLU(0.1)
        self.drop = torch.nn.Dropout()
        self.p1 = torch.nn.MaxPool1d(2)
        self.c2 = torch.nn.Conv1d(20,1,2)
        #torch.nn.LeakyReLU(0.1)
        #torch.nn.Dropout()
        self.p2 = torch.nn.MaxPool1d(2)
        self.linear = torch.nn.Linear(4,4)
        self.linear2 = torch.nn.Linear(4,1)
        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        x = self.c1(input)
        x = self.relu(x)
        x = self.drop(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.relu(x)
        x = self.drop(x)
	print x
	print x.squeeze()
        x = self.p2(x)
        x = self.linear(x)
        x = self.drop(x)
	x = x.view(1,4)
        x = self.linear2(x)
        return self.tanh(x)

'''
loss_fn = torch.nn.MSELoss(size_average=True)

learning_rate = 1e-6

model = Discrim()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for iteration in range(5000):
    mini_batch = data.sample(n = 1,replace = True)
    y_train = mini_batch[mini_batch.columns[20]]
    del mini_batch[mini_batch.columns[20]]
    x_train = mini_batch
    x_train = x_train.values.tolist()[0]

    x_train = np.array(x_train)
    y_train = np.array(y_train.values, dtype=np.float64)    
    inpt_train_x = torch.from_numpy(x_train)
    inpt_train_x = inpt_train_x.float()
    inpt_train_y = torch.from_numpy(y_train)
    inpt_train_y = inpt_train_y.float()
    inpt_train_x = Variable(inpt_train_x)
    inpt_train_y = Variable(inpt_train_y, requires_grad=False)

    y_pred = model(inpt_train_x.view(1,1,20,20))
    loss = loss_fn(y_pred, inpt_train_y)
    if iteration%100 == 0:
        print loss[0].data.numpy()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

total = 0
correct = 0
prediction = -1
for item in test:
	features ,label = item[:20],item[20]
	features = features.tolist()
	#print features
	features = np.array(features)
	label = np.array(label, dtype=np.float64)
	features = torch.from_numpy(features)
	features = features.float()
	features = Variable(features)
	predict = model.forward(features.view(1,1,20,20)).data.numpy()
	
	if predict[0][0] < 5.0:
		prediction = 0.0
	else:
		prediction = 1.0
	if prediction == label:
		correct = correct + 1
	total = total + 1
print correct, correct/total
print "done"
'''