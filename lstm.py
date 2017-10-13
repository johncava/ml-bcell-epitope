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

a = np.array([[[1,1],[6,6],[11,11]],[[2,2],[7,7],[12,12]],
	[[3,3],[8,8],[13,13]],[[4,4],[9,9],[14,14]],[[5,5],[10,10],[15,15]]])
a = torch.from_numpy(a)
a = a.float()
a = Variable(a.view(5,3,2))
print a

lstm = nn.LSTM(2,1,2)
output, _ = lstm(a)
last_output = output[-1]

print output

