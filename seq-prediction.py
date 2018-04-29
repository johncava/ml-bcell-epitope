import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from utility import *

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

# load data
data = load_test('SEQ194.txt')
split = int(len(data)*0.80)
train, test = data[:split], data[split:]
print "Dataset created", len(train)

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.lstm = nn.LSTM(20,2)
		self.sigmoid = nn.Sigmoid()
		self.hidden = self.init_hidden()

	def init_hidden(self):
		return (autograd.Variable(torch.randn(1, 1, 2)),
			autograd.Variable(torch.randn((1, 1, 2))))

	def forward(self,i):
		out, self.hidden = self.lstm(i.view(1, 1, -1), self.hidden)
		return out

def encode_input(x):
	return table_hot[x]

def encode_output(y):
	if y == '0':
		return [1.0,0.0]
	elif y == '1':
		return [0.0,1.0]

model = Model()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-1)

# initialize the hidden state. Keep hidden layer resets out of the training phase (maybe except when testing)
hidden = (autograd.Variable(torch.randn(1, 1, 2)),
         autograd.Variable(torch.randn((1, 1, 2))))

loss = 0

loss_array = []

for epoch in xrange(3):
	#l = 0
	# Note: reset loss such that doesn't accumulate after each epoch
	for sequence in xrange(len(train)):
		inputs = [Variable(torch.Tensor(encode_input(x))) for x in train[sequence][1]]
		outputs = [Variable(torch.Tensor(encode_output(y))).view(1,2).long() for y in train[sequence][2]]		
		loss = 0
		optimizer.zero_grad()
		model.hidden = model.init_hidden()	
		for i, label in zip(inputs,outputs):
			# Step through the sequence one element at a time.
			# after each step, hidden contains the hidden state.
			out = model(i)
			#loss += loss_function(out.view(1,9),label)
			loss += loss_function(out.view(1,2), torch.max(label, 1)[1])
			#l = loss
		loss_array.append(loss[0].data.numpy().tolist())
		#print 'Sequence ', (sequence + 1)
		loss.backward()#retain_graph=True)
		optimizer.step()

plt.plot(xrange(1,len(loss_array) + 1), loss_array)
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.title('Entropy Loss of LSTM with One Hot Encoded lr=1e-3 (5 epochs)')
plt.show()
plt.savefig('result_seq_lr=1e-3_5epochs.png')

print 'Done 1'
torch.save(model.state_dict(), "seq.model")

