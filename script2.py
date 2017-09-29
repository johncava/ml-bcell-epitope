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

# Definitions of functions
def accuracy(error):
    num = 0
    for item in error:
        if item == 0:
            num = num + 1
    return (num/len(error))*100

# Hot Vector Representations for Amino Acids
table_hot = {'A': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         'R': [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         'N': [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         'D': [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         'C': [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         'E': [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         'Q': [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
         'G': [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
         'H': [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
         'I': [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
         'L': [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
         'K': [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
         'M': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
         'F': [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
         'P': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
         'S': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
         'T': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
         'W': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
         'Y': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
         'V': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}

# Float representations for Amino Acids
table_float = {'A':1.0,
         'R':2.0,
         'N':3.0,
         'D':4.0,
         'C':5.0,
         'E':6.0,
         'Q':7.0,
         'G':8.0,
         'H':9.0,
         'I':10.0,
         'L':11.0,
         'K':12.0,
         'M':13.0,
         'F':14.0,
         'P':15.0,
         'S':16.0,
         'T':17.0,
         'W':18.0,
         'Y':19.0,
         'V':20.0}

# Positive and Negative Datasets
pos_data = []
neg_data = []

# Read in the Positive Dataset
with open('pos.data') as f:
    for line in f:
        l = []
        line = line.rstrip()
        for AA in line:
            l.append(table_hot[AA])
        l.append(1.0)
        pos_data.append(l)

# Read in the Negative Dataset
with open('neg.data') as f:
    for line in f:
        l = []
        line = line.rstrip()
        for AA in line:
            l.append(table_hot[AA])
        l.append(-1.0)
        neg_data.append(l)


# Data preparation
data = pos_data + neg_data
data = np.array(data)
np.random.shuffle(data)
data, test = np.array_split(data,2)
print len(data), len(test)
data = pd.DataFrame(data)
'''
print data.sample(frac=0.008,replace = True)
y_train = data[data.columns[20]]
del data[data.columns[20]]
x_train = data
x_train = np.array(x_train.as_matrix(), dtype=np.float64)
y_train = np.array(y_train.values, dtype=np.float64)
'''
#print data
test = pd.DataFrame(test)
y_test = test[test.columns[20]]
del test[test.columns[20]]
x_test = test
x_test = np.array(x_test.as_matrix(), dtype=np.float64)
y_test = np.array(y_test.values, dtype=np.float64)

# Vanilla Deep Learning Model
'''
model = torch.nn.Sequential(
    torch.nn.Conv1d(1,20,3),
    torch.nn.LeakyReLU(0.1),
    torch.nn.Dropout(),
    torch.nn.MaxPool1d(2),
    torch.nn.Conv1d(20,1,2),
    torch.nn.LeakyReLU(0.1),
    torch.nn.Dropout(),
    torch.nn.MaxPool1d(2),
)
'''

class Discrim(nn.Module):
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
        self.tanh = torch.nn.TanH()

    def forward(self, input):
        x = self.c1(input)
        x = self.relu(x)
        x = self.drop(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.p2(x)
        x = self.linear(x)
        x = self.drop(x)
	x = x.view(1,4)
        x = self.linear2(x)
        return self.tanh(x)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 2e-4

model = Discrim()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for iteration in range(800):
    mini_batch = data.sample(n = 1,replace = True)
    y_train = mini_batch[mini_batch.columns[20]]
    del mini_batch[mini_batch.columns[20]]
    x_train = mini_batch
    x_train = np.array(x_train.as_matrix(), dtype=np.float64)
    y_train = np.array(y_train.values, dtype=np.float64)
    
    inpt_train_x = torch.from_numpy(x_train)
    inpt_train_x = inpt_train_x.float()
    inpt_train_y = torch.from_numpy(y_train)
    inpt_train_y = inpt_train_y.float()
    inpt_train_x = Variable(inpt_train_x)
    inpt_train_y = Variable(inpt_train_y, requires_grad=False)
    
    y_pred = model(inpt_train_x.view(1,20,20))
    loss = loss_fn(y_pred, inpt_train_y)
    if iteration%100 == 0:
        print loss[0].data.numpy()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

inpt_test_x = torch.from_numpy(x_test)
inpt_test_x = inpt_test_x.float()
inpt_test_x = Variable(inpt_test_x)

'''
array = model.forward(inpt_test_x).data.numpy()
new_array = []
for index in array:
    if index[0] >= 0.0:
        new_array.append(1.0)
    if index[0] < 0.0:
        new_array.append(-1.0)
new_array = np.array(new_array)
error = y_test - new_array
a,b = calculate_roc(new_array.tolist(),y_test.tolist())
print a,b
#print metrics.auc([b],[a])
deep_acc = accuracy(error)
print 'Vanilla Deep Learning Model Accuracy: ' ,deep_acc, '%'
'''
print "done"
