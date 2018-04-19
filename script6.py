from __future__ import division
from utility import *
import torch
import random
from torch.autograd import Variable

test_data = load_test('SEQ194.txt')
#print test[0]

protvec = initialize()

# Positive and Negative Datasets
pos_data = []
neg_data = []

# Read in the Positive Dataset
with open('corpus_pos.txt') as f:
    for line in f:
        line = line.rstrip()
        p = embedding(protvec,line)
        pos_data.append([p,1.0])

# Read in the Negative Dataset
with open('corpus_neg.txt') as f:
    for line in f:
        n = []
        line = line.rstrip()
        n = embedding(protvec, line)
        neg_data.append([n,0.0])

# Data preparation
data = pos_data + neg_data
data = np.array(data)
np.random.shuffle(data)

data = data.tolist()
#x,y = data[0][0], data[0][1]
#print x
#iprint y
#input = Variable(torch.from_numpy(np.array(x))).view(1,3,100).double()
#print input

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
        self.linear2 = torch.nn.Linear(12,12).double()
        self.linear3 = torch.nn.Linear(12,1).double()
	self.tanh = torch.nn.Tanh()
	self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        x = self.c1(input)
        x = self.relu(x)
        #x = self.drop(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.relu(x)
        #x = self.drop(x)
        x = self.p2(x)
        x = self.linear(x.view(1,24))
        x = self.relu(x)
#x = self.drop(x)
        x = self.linear2(x)
	x = self.relu(x)
	x = self.linear3(x)
        return self.sigmoid(x)

loss_fn = torch.nn.MSELoss(size_average=True)

learning_rate = 1e-4

model = Discrim()

#print len(data) --> 633

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#print data[0]

for epoch in xrange(10):
	for index in xrange(len(data)):
		#print index
		train = data[index]
		x , y = train[0] , train[1]
		#print x,y
		#break
		x = Variable(torch.from_numpy(np.array(x)), requires_grad = False).view(1,3,100).double()

		inpt_train_y = torch.from_numpy(np.array([[y]]))
		inpt_train_y = inpt_train_y.double()
		inpt_train_y = Variable(inpt_train_y, requires_grad=False)
		#print inpt_train_y
		y_pred = model(x)
		loss = loss_fn(y_pred, inpt_train_y)
		if index%100 == 0:
			print loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

# Testing

window = 20
for test in test_data:
	sequence, label = test[1], test[2]
	# Create window lists
	prediction = [[]] * len(sequence)
	indices = range(len(sequence))
	indices_list = []
	for i in xrange(len(sequence) - window):
		a = indices[i:i+window]
		indices_list.append(a)
		# Predict
		embed = embedding(protvec, sequence[i:i+window])
		#print embed
		x = Variable(torch.from_numpy(np.array(embed)), requires_grad = False).view(1,3,100).double()
		pred = model(x)
		#print pred[0][0].data.numpy()
		for index in a:
			prediction[index].append(pred[0][0].data.numpy().tolist())
		break
	for index, p in enumerate(prediction):
		prediction[index] = sum(p) / float(len(p))
	print prediction
