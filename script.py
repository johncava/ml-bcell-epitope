
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

# Positive and Negative Datasets
pos_data = []
neg_data = []

# Read in the Positive Dataset
with open('pos.data') as f:
    for line in f:
        l = []
        line = line.rstrip()
        for AA in line:
            l.append(table[AA])
        pos_data.append(l)

# Read in the Negative Dataset
with open('neg.data') as f:
    for line in f:
        l = []
        line = line.rstrip()
        for AA in line:
            l.append(table[AA])
        neg_data.append(l)

print len(pos_data)
print len(neg_data)