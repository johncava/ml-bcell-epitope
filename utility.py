from __future__ import division
from split import chop
import numpy as np
import pickle 

def calculate_roc(predict, true):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	print predict
	print true
	print len(predict), len(true)
	sensitivity, specificity = 0,0
	print predict[1]
	for i in xrange(len(predict)):
		print i
		if predict[i] == 1.0 and true[i] == 1.0:
			TP = TP + 1
		elif predict[i] == 1.0 and true[i] == -1.0:
			FP = FP + 1
		elif predict[i] == -1.0 and true[i] == -1.0:
			TN = TN + 1
		elif predict[i] == -1.0 and true[i] == 1.0:
			FN = FN + 1
	print "done"
	sensitivity = TP/(TP + FN)
	specificity = TN/(FP + TN)
	return sensitivity, specificity

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def initialize():
    return load_obj('protVec')

def split(start, model, seq, lis):
    for index in xrange(start,len(seq) - 2,3):
        kmer = seq[index:index+3].encode('utf-8')
        if kmer in model:
            lis.append(np.array(model[kmer]))
        else:
            lis.append(np.array(model['<unk>']))
    lis = np.mean(lis, axis=0).tolist()
    return lis   

def embedding(model, seq):
    first, second, third = [], [] , []
    #First Split
    first = split(0, model, seq, first)
    #Second Split
    second = split(1, model, seq, second)
    #Third Split
    third = split(2, model, seq, third)
    return [first, second, third]

def load_test(data):
    array = []
    with open(data, 'r') as file:
       for line in file:
          line = line.strip('\n')
          array.append(line)
    return list(chop(3,array))

