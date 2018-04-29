from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

aap_table = {
	'A': [2.0,1.0,2.0,1.0],
	'R': [1.0,3.0,3.0,3.0],
	'N': [1.0,2.0,3.0,2.0],
	'D': [1.0,1.0,3.0,1.0],
	'C': [3.0,1.0,1.0,2.0],
	'E': [1.0,2.0,3.0,2.0],
	'Q': [1.0,2.0,3.0,2.0],
	'G': [2.0,1.0,2.0,1.0],
	'H': [2.0,3.0,3.0,3.0],
	'I': [3.0,2.0,1.0,2.0],
	'L': [3.0,2.0,1.0,2.0],
	'K': [1.0,3.0,3.0,3.0],
	'M': [3.0,3.0,1.0,3.0],
	'F': [3.0,3.0,1.0,3.0],
	'P': [2.0,1.0,2.0,2.0],
	'S': [2.0,1.0,2.0,1.0],
	'T': [2.0,1.0,2.0,1.0],
	'W': [3.0,3.0,1.0,3.0],
	'Y': [2.0,3.0,1.0,3.0],
	'V': [3.0,2.0,1.0,2.0]
}

pos_data = []
neg_data = []

# Read in the Positive Dataset
with open('pos.data') as f:
    for line in f:
        l = []
        line = line.rstrip()
        for AA in line:
            l.append(aap_table[AA])
        l.append(1.0)
        pos_data.append(l)

# Read in the Negative Dataset
with open('neg.data') as f:
    for line in f:
        l = []
        line = line.rstrip()
        for AA in line:
            l.append(aap_table[AA])
        l.append(0.0)
        neg_data.append(l)

#Data preparation
data = pos_data + neg_data
data = np.array(data)
np.random.shuffle(data)
data, test = np.array_split(data,2)
data = data.tolist()

iterations = 1000
lstm = nn.LSTM(4,1,15)
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4

sigmoid = nn.Sigmoid()
optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

for iteration in xrange(iterations):
        num = np.random.choice(len(data),1)

        train_x, train_y = data[num[0]][:20], data[num[0]][-1]
        train_x = np.array(train_x)
        train_x = torch.from_numpy(train_x)
        train_x = train_x.float()
        train_x = Variable(train_x)
        train_x = train_x.view(20,1,4)

        train_y = np.array([train_y])
        train_y = torch.from_numpy(train_y)
        train_y = train_y.float()
        train_y = Variable(train_y)

        output, _ = lstm(train_x)
        output = sigmoid(output)
        last_output = output[-1]
        loss = loss_fn(last_output, train_y)
        if iteration%100 == 0:
                print loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

total = 0
correct = 0
prediction = -1
test = test.tolist()
for item in xrange(len(test)):
        features ,label = test[item][:20], test[item][-1]
        features = np.array(features)
        label = np.array(label, dtype=np.float64)
        features = torch.from_numpy(features)
        features = features.float()
        features = Variable(features)
        predict, _ = lstm(features.view(20,1,4))
        predict = sigmoid(predict)
        predict =  predict[-1][0][0].data[0]
        if predict < 0.5:
                prediction = 0.0
        else:
                prediction = 1.0
        if prediction == label:
                correct = correct + 1
        total = total + 1
print correct, correct/total, total
