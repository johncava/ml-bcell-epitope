from __future__ import division
from utility import *
import torch
import random
from torch.autograd import Variable
import torch.nn.functional as F

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
        pos_data.append([p,[1.0,0.0]])

# Read in the Negative Dataset
with open('corpus_neg.txt') as f:
    for line in f:
        n = []
        line = line.rstrip()
        n = embedding(protvec, line)
        neg_data.append([n,[0.0,1.0]])

# Data preparation
data = pos_data + neg_data
data = np.array(data)
np.random.shuffle(data)

data1, data2, test = np.array_split(data,3)

data = np.array(data1.tolist() + data2.tolist())
data = data.tolist()
#x,y = data[0][0], data[0][1]
#print x
#iprint y
#input = Variable(torch.from_numpy(np.array(x))).view(1,3,100).double()
#print input

# 1D convolution
class Alpha(torch.nn.Module):
    	def __init__(self):
        	super(Alpha, self).__init__()
		self.view1 = torch.nn.Sequential(
			torch.nn.Linear(100,100),
			#torch.nn.Dropout(),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(100,50),
			#torch.nn.Dropout(),
			torch.nn.LeakyReLU()
		)

		self.view2 = torch.nn.Sequential(
			torch.nn.Linear(100,100),
			#torch.nn.Dropout(),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(100,50),
			#torch.nn.Dropout(),
			torch.nn.LeakyReLU()
		)

		self.view3 = torch.nn.Sequential(
			torch.nn.Linear(100,100),
			#torch.nn.Dropout(),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(100,50),
			#torch.nn.Dropout(),
			torch.nn.LeakyReLU()
		)

		self.fc = torch.nn.Sequential(
			torch.nn.Linear(50,50),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(50,50),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(50,2)
		)

	def forward(self, a, b, c):
		x = self.view1(a)
		y = self.view2(b)
		z = self.view3(c)
		connect = torch.add(torch.add(x,y), z)
		ans = self.fc(connect)
		return F.softmax(ans, dim=1)

loss_fn = torch.nn.MSELoss(size_average=True)

learning_rate = 1e-4

model = Alpha()

#print len(data) --> 633

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#print data[0]

batch = 5

for epoch in xrange(10):
	for index in xrange(int(len(data)/batch)):
		#print index
		data_batch = data[index*batch:index*batch + batch]
		#print x,y
		#break
		#x = Variable(torch.from_numpy(np.array(x)), requires_grad = False).view(1,3,100).double()
		a = []
		b = []
		c = []
        	y = []
        	for train_x, train_y in data_batch:
                	a.append(train_x[0])
			b.append(train_x[1])
			c.append(train_x[2])
                	y.append(train_y)
		
		a = np.array(a)
		a = torch.from_numpy(a)
		a = Variable(a, requires_grad=False)
		a = a.float()
		
		b = np.array(b)
		b = torch.from_numpy(b)
		b = Variable(b, requires_grad=False)
		b = b.float()
		
		c = np.array(c)
		c = torch.from_numpy(c)
		c = Variable(c, requires_grad=False)
		c = c.float()
		
		inpt_train_y = torch.from_numpy(np.array(y))
		inpt_train_y = inpt_train_y.float()
		inpt_train_y = Variable(inpt_train_y, requires_grad=False)
		
		#print inpt_train_y
		y_pred = model(a,b,c)
		#print y_pred
		#break
		loss = loss_fn(y_pred, inpt_train_y)
		#if index%10 == 0:
			#print '--------'
			#print y_pred[0][0].data.numpy().tolist(), y_pred[0][1].data.numpy().tolist()
			#print inpt_train_y
			#print loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	#break
# Testing


total = 0
correct = 0
for item in xrange(len(test)):
	features ,label = test[item][0], test[item][1]
	#features = np.array(features)
	label = np.array(label, dtype=np.float64)
	features1 = torch.from_numpy(np.array([features[0]]))
	features1 = features1.float()
	features1 = Variable(features1)

	features2 = torch.from_numpy(np.array([features[1]]))
	features2 = features2.float()
	features2 = Variable(features2)

	features3 = torch.from_numpy(np.array([features[2]]))
	features3 = features3.float()
	features3 = Variable(features3)
	predict = model.forward(features1,features2,features3).data.numpy()

	class1, class2 = y_pred[0][0].data.numpy().tolist(), y_pred[0][1].data.numpy().tolist()
	#print class1, class2
	class_ = None
	#print label
	if class1 > class2:
		class_ = [1.0,0.0]
	else:
		class_ = [0.0,1.0]
	if class_ == label.tolist():
		correct = correct + 1
	total = total + 1
print correct, correct/total, total


'''
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
		x = Variable(torch.from_numpy(np.array([embed[0]])), requires_grad = False).float()
		y = Variable(torch.from_numpy(np.array([embed[1]])), requires_grad = False).float()
		z = Variable(torch.from_numpy(np.array([embed[2]])), requires_grad = False).float()
		pred = model(x,y,z)
		#print pred[0][0].data.numpy()
		class1, class2 = y_pred[0][0].data.numpy().tolist(), y_pred[0][1].data.numpy().tolist()
		#print class1, class2
		#break
		class_ = None
		if class1 > class2:
			class_ = 1.0
			#print 'hi'
		else:
			class_ = 0.0
			#print 'hey'	
		for index in a:
			prediction[index].append(class_)
	#print prediction
	#break
	for index, p in enumerate(prediction):
		prediction[index] = sum(p) / float(len(p))
	print prediction
'''