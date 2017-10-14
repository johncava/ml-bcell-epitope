import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
'''
a = np.array([[1,6,11],[2,7,12],[3,8,13],[4,9,14],[5,10,15]])
a = torch.from_numpy(a)
a = a.float()
a = Variable(a.view(5,3,1))
print a

lstm = nn.LSTM(1,1,2)
output, _ = lstm(a)
last_output = output[-1]

print output
'''
'''
a = np.array([[[1,1],[6,6],[11,11]],[[2,2],[7,7],[12,12]],
	[[3,3],[8,8],[13,13]],[[4,4],[9,9],[14,14]],[[5,5],[10,10],[15,15]]])
a = torch.from_numpy(a)
a = a.float()
a = Variable(a.view(5,3,2))
print a
'''
# Hot Vector Representations for Amino Acids
table_hot = {'A': [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'R': [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'N': [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'D': [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'C': [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'E': [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'Q': [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'G': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'H': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'I': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'L': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'K': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'M': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'F': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
             'P': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
             'S': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],
             'T': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],
             'W': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
             'Y': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
             'V': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]}

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
        l.append(0.0)
        neg_data.append(l)

#Data preparation
data = pos_data + neg_data
data = np.array(data)
np.random.shuffle(data)
data, test = np.array_split(data,2)
data = data.tolist()
'''
num = np.random.choice(len(data),1)

train_x, train_y = data[num[0]][:20], data[num[0]][-1]
train_x = np.array(train_x)
train_x = torch.from_numpy(train_x)
train_x = train_x.float()
train_x = Variable(train_x)
train_x = train_x.view(20,1,20)

train_y = np.array([train_y])
train_y = torch.from_numpy(train_y)
train_y = train_y.float()
train_y = Variable(train_y)
'''
iterations = 100
lstm = nn.LSTM(20,1,2)
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4

optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

for iteration in xrange(iterations):
	num = np.random.choice(len(data),1)
	
	train_x, train_y = data[num[0]][:20], data[num[0]][-1]
	train_x = np.array(train_x)
	train_x = torch.from_numpy(train_x)
	train_x = train_x.float()
	train_x = Variable(train_x)
	train_x = train_x.view(20,1,20)

	train_y = np.array([train_y])
	train_y = torch.from_numpy(train_y)
	train_y = train_y.float()
	train_y = Variable(train_y)

	output, _ = lstm(train_x)
	last_output = output[-1]
	loss = loss_fn(last_output, train_y)
	print loss.data[0]
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()	
