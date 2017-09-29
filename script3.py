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
        p = embedding(protvec,line)
        pos_data.append([p,1.0])

# Read in the Negative Dataset
with open('neg.data') as f:
    for line in f:
        n = []
        line = line.rstrip()
        n = embedding(protvec, line)
        neg_data.append([n,-1.0])


# Data preparation
data = pos_data + neg_data
data = np.array(data)
np.random.shuffle(data)
data, test = np.array_split(data,2)
print len(data), len(test)

data = data.tolist()
x,y = data[0][0], data[0][1]
print x
print y
input = Variable(torch.from_numpy(np.array(x))).view(1,3,100).double()
print input

# 1D convolution
class Discrim(torch.nn.Module):
    def __init__(self):
        super(Discrim, self).__init__()
        self.c1 = torch.nn.Conv1d(3,20,3).double()
        self.relu = torch.nn.LeakyReLU(0.1)
        self.drop = torch.nn.Dropout()
        self.p1 = torch.nn.MaxPool1d(2)
        self.c2 = torch.nn.Conv1d(20,1,2).double()
        #torch.nn.LeakyReLU(0.1)
        #torch.nn.Dropout()
        self.p2 = torch.nn.MaxPool1d(2)
        self.linear = torch.nn.Linear(24,12).double()
        self.linear2 = torch.nn.Linear(12,1).double()
        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        x = self.c1(input)
        x = self.relu(x)
        x = self.drop(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.p2(x)
        x = self.linear(x.view(1,24))
        x = self.drop(x)
        x = self.linear2(x)
        return self.tanh(x)

loss_fn = torch.nn.MSELoss(size_average=True)

learning_rate = 1e-4

model = Discrim()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for iteration in range(1000):

    inpt_train_y = torch.from_numpy(np.array([y]))
    inpt_train_y = inpt_train_y.double()
    inpt_train_y = Variable(inpt_train_y, requires_grad=False)
    print inpt_train_y
    y_pred = model(input)
    loss = loss_fn(y_pred, inpt_train_y)
    if iteration%100 == 0:
        print loss[0].data.numpy()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

total = 0
correct = 0
prediction = -1
test = test.tolist()
for item in xrange(len(test)):
	features ,label = test[item][0], test[item][1]
	features = np.array(features)
	label = np.array(label, dtype=np.float64)
	features = torch.from_numpy(features)
	features = features.float()
	features = Variable(features)
	predict = model.forward(features.view(1,3,100).double()).data.numpy()
	
	if predict[0][0] < 0.0:
		prediction = -1.0
		#print "hi"
	else:
		prediction = 1.0
		#print "world!"
	if prediction == label:
		correct = correct + 1
	total = total + 1
print correct, correct/total, total
print "done"
