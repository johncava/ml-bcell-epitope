from __future__ import division
from sklearn import svm
import numpy as np
import pandas as pd
import scipy as scip
from scipy import linalg
from scipy.spatial.distance import pdist, squareform

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
np.random.shuffle(data)
data = pd.DataFrame(data)
y_train = data[data.columns[20]]
del data[data.columns[20]]
x_train = data
x_train = np.array(x_train .as_matrix(), dtype=np.float64)
y_train = np.array(y_train.values, dtype=np.float64)

# SVM RBF Training 
clf = svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)
ans = clf.predict(x_train)
error =  y_train - ans
print 'Sci-kit Learn SVM Vanilla Model: ', accuracy(error)

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

###
# Notes:
#   Support Vector Machine trained by gradient descent can match svm from sklearn
#   However may be less accurate
###

