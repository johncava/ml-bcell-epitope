import torch
import torch.nn as nn
from torch.autograd import Variable

c1 = nn.Conv2d(1,10,3)
p1 = nn.MaxPool2d(2)
c2 = nn.Conv2d(10,10,2)
p2 = nn.MaxPool2d(2)
c3 = nn.Conv2d(10,1,1)
p3 = nn.MaxPool2d(2)
linear = nn.Linear(4,1)

input = Variable(torch.randn(20,20))
conv = p3(c3(p2(c2(p1(c1(input.view(1,1,20,20)))))))
print conv
print linear(conv.view(1,4))
