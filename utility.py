from __future__ import division

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
