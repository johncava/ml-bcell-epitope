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

# Float representations for Amino Acids
table = {'A':1.0,
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
            l.append(table[AA]/20.0)
        l.append(1.0)
        pos_data.append(l)

# Read in the Negative Dataset
with open('neg.data') as f:
    for line in f:
        l = []
        line = line.rstrip()
        for AA in line:
            l.append(table[AA]/20.0)
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

'''
# SVM RBF Training 
clf = svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)
ans = clf.predict(x_test)
error =  y_test - ans
print 'Sci-kit Learn SVM Vanilla Model: ', accuracy(error)
'''

'''
# SVM  RBF hyperparameters
sigma = 1.0
length = 1.0

noise = 0.00001

# SVM computed gram matrix
clf = svm.SVC(kernel='precomputed')
pairwise_dists = squareform(pdist(x_train, 'euclidean'))
for iteration in xrange(20):
    my_kernel = (sigma ** 2)*scip.exp(-(pairwise_dists ** 2) / (2 * length ** 2))
    k_inv = np.linalg.inv(my_kernel + noise*np.identity(len(my_kernel)))
    
    d_my_kernel = ((pairwise_dists ** 2)/ (length ** 3))*(sigma ** 2)*scip.exp(- (pairwise_dists ** 2) / (2 *length ** 2))
    length_er = 0.5*np.trace(np.dot(k_inv,d_my_kernel)) - 0.5*np.dot(np.dot(np.dot(np.dot(y_train.T,k_inv),d_my_kernel),k_inv),y_train)

    d_my_kernel = (2*sigma)*scip.exp(- (pairwise_dists ** 2) / (2*length ** 2))
    sigma_er = 0.5*np.trace(np.dot(k_inv,d_my_kernel)) - 0.5*np.dot(np.dot(np.dot(np.dot(y_train.T,k_inv),d_my_kernel),k_inv),y_train)

    sigma = sigma - sigma_er
    length = length - length_er
    
    print sigma
    print length

my_kernel = (sigma ** 2) * scip.exp(-(squareform(pdist(x_train, 'euclidean')) ** 2) / (2 *length ** 2))
clf.fit(my_kernel, y_train)
ans = clf.predict(my_kernel)
error =  y_train - ans
print 'SVM trained by gradient descent: ', accuracy(error)
print 'Done'
'''

###
# Notes:
#   Support Vector Machine trained by gradient descent can match svm from sklearn
#   However may be less accurate
###
'''
inpt_train_x = torch.from_numpy(x_train)
inpt_train_x = inpt_train_x.float()
inpt_train_y = torch.from_numpy(y_train)
inpt_train_y = inpt_train_y.float()

inpt_train_x = Variable(inpt_train_x)
inpt_train_y = Variable(inpt_train_y, requires_grad=False)
'''
# Vanilla Deep Learning Model
model = torch.nn.Sequential(
    torch.nn.Linear(20, 40),
    torch.nn.LeakyReLU(0.1),
    torch.nn.Linear(40,40),
    torch.nn.LeakyReLU(0.1),
    torch.nn.Linear(40,1),
    torch.nn.Tanh()   
)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 2e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for iteration in range(800):
    mini_batch = data.sample(frac=0.05,replace = True)
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
    
    y_pred = model(inpt_train_x)
    loss = loss_fn(y_pred, inpt_train_y)
    '''
    if iteration%100 == 0:
        print loss
    '''
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


inpt_test_x = torch.from_numpy(x_test)
inpt_test_x = inpt_test_x.float()
inpt_test_x = Variable(inpt_test_x)

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
