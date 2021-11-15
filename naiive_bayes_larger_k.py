#get the k-mer frequencies for the given sequences
#aim: get the most discriminative k-mers

#try k = 4,32, 64, 128

#train an NB classifier using sklearn:
#https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes

from KmerConstruct import KmerConstruct

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

#read sequences after trimming 

#ss=['atcg','gatc','tacg']
ss=['atcgatat','cgatcgcg','atcgcgat','gatcgaga','gatctctc','gatcgatc','tacgcgcg','tatatacg','cgtatacg']
y=[0,0,0,1,1,1,2,2,2]
k=4


#larger test data
import random
ss=[]
y=[]
for i in range(100):
    ss.append(''.join(random.choices('atcg',k=1000))) #generate a random sequence of length k
    y.append(random.choices([0,1,2],k=1)) #generate random labels
y=np.array(y)

k=10


#real data from silva

#read fasta files and return 2 lists: labels, sequences
def readFasta(filename):
    labels=[]
    seqs = []
    with open(filename, 'r') as f:
        for line in f:
            line=line.rstrip()
            if line[0] == '>':
                labels.append(line[1:])
            else:
                seqs.append(line)
    return labels, seqs 

os.chdir('/Users/ifanwu/Documents/CS_MS_TAMU/microbiota_ML')
label_silva_138_99_V4,seq_silva_138_99_V4=readFasta('silva/silva138_99_truncated.fasta') 
ss=seq_silva_138_99_V4
k=100

#read taxonomy data
import pandas as pd
taxonomy_table=pd.read_csv('silva/taxonomy.tsv',sep='\t')
labels=taxonomy_table['Taxon'].tolist()


#get genus as labels
genus_labels=[]
for label in labels:
    genus=label.split(';')[-2][1:]
    genus_labels.append(genus) #[1:] is to strip off the first space  #-1:species, -2:genus,....etc.
    if genus.find('g__')==-1:
        print("this is found:",genus)
        raise ValueError("No species label found!")
genus_labels=np.array(genus_labels)


##sample a small subset of the data

from numpy.random import default_rng

size=1000
rng = default_rng()
indices = rng.choice(len(ss), size=size, replace=False)
ss_small=[ss[i] for i in indices]
genus_labels_small=genus_labels[indices]
k=10



#for each sequences, construct the frequency table 
new_KmerConstruct=KmerConstruct(ss_small,k)
new_KmerConstruct.constuct_feature_table()
#new_KmerConstruct.constuct_feature_table(j_=5)
X=new_KmerConstruct.X
y=genus_labels_small
X_first_j=new_KmerConstruct.X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print(accuracy_score(y_pred,y_test))
