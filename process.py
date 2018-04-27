import pickle
import numpy as np
import random

# Float representations for Amino Acids
table = {'A':1.0,
         'R':2.0,
         'N':3.0,
         'D':4.0,
         'C':5.0,
         'E':6.0,
         'Q':7.0,
         'G':8.0,
         'H':9.0,
         'I':10.0,
         'L':11.0,
         'K':12.0,
         'M':13.0,
         'F':14.0,
         'P':15.0,
         'S':16.0,
         'T':17.0,
         'W':18.0,
         'Y':19.0,
         'V':20.0}

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
    return first, second, third

def create_data():
    data = []
    with open('pos.data') as file:
        for line in file:
            line = line.rstrip('\n')
            line = line.split()
            epitope = []
            for aa in line[0]:
                epitope.append(table[aa])
            #print epitope
            data.append(epitope)
    return data

def sample_data(data):
    return np.array([random.choice(data)])

def create_fake_data():
    epitope = []
    for index in xrange(20):
        epitope.append(random.random())
    return np.array([epitope])

data = create_data()
print sample_data(data)
'''
model = initialize()
num = 0
batch = []
batch_size = 3
with open('pos.data') as file:
    for line in file:
        line = line.rstrip('\n')
        line = line.split()
        f,s,t = embedding(model, line[0])
        mean = (np.array(f) + np.array(s) + np.array(t))/3.0
        #print mean
        num += 1
        batch.append(mean)
        if num > 10:
            break

print random.sample(batch, batch_size)
'''
