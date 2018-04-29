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

# Vector Representations for Amino Acids
# Amino Acid: [Charge (-1/0/1), Hydropath index, Van der Waals Volume (A^3), Polar/Nonpolar(1/-1)]
table_hot = {'A': [0.0,1.8,67.0,-1.0],
             'R': [1.0,-4.5,148.0,1.0],
             'N': [0.0,-3.5,96.0,1.0],
             'D': [-1.0,-3.5,91.0,1.0],
             'C': [0.0,2.5,86.0,-1.0],
             'E': [-1.0,-3.5,109.0,1.0],
             'Q': [0.0,-3.5,114.0,1.0],
             'G': [0.0,-0.4,48.0,-1.0],
             'H': [1.0,-3.2,118.0,1.0],
             'I': [0.0,4.5,124.0,-1.0],
             'L': [0.0,3.8,124.0,-1.0],
             'K': [1.0,-3.9,135.0,1.0],
             'M': [0.0,1.9,124.0,-1.0],
             'F': [0.0,2.8,135.0,-1.0],
             'P': [0.0,-1.6,90.0,-1.0],
             'S': [0.0,-0.8,73.0,1.0],
             'T': [0.0,-0.7,93.0,1.0],
             'W': [0.0,-0.9,163.0,-1.0],
             'Y': [0.0,-1.3,141.0,1.0],
             'V': [0.0,4.2,105.0,-1.0]
             }

# load data
data = load_test('SEQ194.txt')
split = int(len(data)*0.80)
train, test = data[:split], data[split:]
print "Dataset created", len(train)

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.lstm = nn.LSTM(4,2)
		self.lstm2 = nn.LSTM(2,2)
		self.linear = nn.Linear(4,4)
		self.linear2 = nn.Linear(4,4)
		self.linear3 = nn.Linear(2,2)
		self.hidden = self.init_hidden()
		self.hidden2 = self.init_hidden()

	def init_hidden(self):
		return (autograd.Variable(torch.zeros(1, 1, 2)),
			autograd.Variable(torch.zeros((1, 1, 2))))

	def forward(self,i):
		i = self.linear(i)
		i = self.linear(i)
		out, self.hidden = self.lstm(i.view(1, 1, -1), self.hidden)
		out2, self.hidden2 = self.lstm2(out.view(1,1,-1), self.hidden2)
		out2 = self.linear3(out2)
		out2 = F.softmax(out2)
		return out2

def encode_input(x):
	return table_hot[x]

def encode_output(y):
	if y == '0':
		return [1.0,0.0]
	elif y == '1':
		return [0.0,1.0]

model = Model()
'''
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss = 0

loss_array = []

for epoch in xrange(3):
	#l = 0
	# Note: reset loss such that doesn't accumulate after each epoch
	for sequence in xrange(len(train)):
		#print train[sequence][0]
		inputs = [Variable(torch.Tensor(encode_input(x))) for x in train[sequence][1]]
		outputs = [Variable(torch.Tensor(encode_output(y))).view(1,2).long() for y in train[sequence][2]]		
		loss = 0
		optimizer.zero_grad()
		model.hidden = model.init_hidden()	
		model.hidden2 = model.init_hidden()
		for i, label in zip(inputs,outputs):
			# Step through the sequence one element at a time.
			# after each step, hidden contains the hidden state.
			out = model(i)
			#loss += loss_function(out.view(1,2),label)
			loss += loss_function(out.view(1,2), torch.max(label, 1)[1])
			#l = loss
		loss_array.append(loss[0].data.numpy().tolist())
		#print loss_array[-1]
		#print 'Sequence ', (sequence + 1)
		loss.backward()#retain_graph=True)
		optimizer.step()

#np.save('lstm1_loss.npy',loss_array)

print 'Done'
torch.save(model.state_dict(), "seq-linear-2-aap.model")


plt.plot(xrange(1,len(loss_array) + 1), loss_array)
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.title('Entropy Loss of Linear and LSTM (2 Layer) with AA Vectorization lr=1e-3')
plt.show()
plt.savefig('result_seq_linear_2_aap_lr=1e-3.png')

'''

# Testing

model.load_state_dict(torch.load('seq-linear-2-aap.model'))

for sequence in xrange(len(test)):
        inputs = [Variable(torch.Tensor(encode_input(x))) for x in test[sequence][1]]
        output = [Variable(torch.Tensor(encode_output(y))) for y in test[sequence][2]]
        model.hidden = model.init_hidden()
        model.hidden2 = model.init_hidden()
	accuracy = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i, label in zip(inputs, output):
                prediction = model(i).view(1,2)
                predict = torch.max(prediction,1)[1].data.numpy().tolist()[0]
                true = torch.max(label,0)[1].data.numpy().tolist()
                #if predict == 1.0:
                #       print 'Hello World!'
                #print true
                if predict == 1 and true == 1:
                        TP = TP + 1
                elif predict == 1 and true == 0:
                        FP = FP + 1
                elif predict == 0 and true == 0:
                        TN = TN + 1
                elif predict == 0 and true == 1:
                        FN = FN + 1
        sensitivity = TP/float(TP + FN)
        specificity = TN/float(FP + TN)
        print (TP,FP,TN,FN)

