#get the k-mer frequencies for the given sequences
#aim: get the most discriminative k-mers, train an multinomial NB classifier using sklearn:


import os
#os.chdir('/Users/ifanwu/Documents/CS_MS_TAMU/microbiota_ML/github') #macos only

from utility import readFasta, get_label_list, get_accu_f1
from KmerConstruct import KmerConstruct

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

import pickle



#load data from silva

#there are sequences lost after truncation. Retrieve the id indices and applied that on the taxonomy labels
label_silva_138_99_V4_truncated,seq_silva_138_99_V4_truncated=readFasta('../silva/silva138_99_truncated.fasta') 
label_silva_138_99_V4,seq_silva_138_99_V4=readFasta('../silva/silva138_99.fasta') 

indices_after_truncation=[]
curr_index=0
for i in range(len(label_silva_138_99_V4_truncated)):
    while label_silva_138_99_V4_truncated[i]!=label_silva_138_99_V4[curr_index]:
        curr_index+=1
    indices_after_truncation.append(curr_index)    
            
    
ss=seq_silva_138_99_V4_truncated


#read taxonomy data
import pandas as pd
taxonomy_table=pd.read_csv('../silva/taxonomy.tsv',sep='\t')
labels=taxonomy_table['Taxon'].tolist()


#get genus as labels
genus_labels=get_label_list(labels,type_='g__')
genus_labels=np.array(genus_labels)
genus_labels=genus_labels[indices_after_truncation]



if __name__ == '__main__':

    ks=[4,6,7,8,9,10,11,12,14,16,18,32,64,100]
    
    
    #construct kmer feature table + train MB model
    
    for k in ks:
        #construct kmer feature table
        new_KmerConstruct=KmerConstruct(ss,k,n_threads=7)
        new_KmerConstruct.constuct_feature_table()
        with open('../KmerConstruct_'+ str(k) + 'mer.pickle', 'wb') as handle:
            pickle.dump(new_KmerConstruct, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        #train MB model
        X=new_KmerConstruct.X
        y=genus_labels
        model = MultinomialNB()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        print("Training error (based on accuracy) is: ",accuracy_score(y_pred,y))

        with open('../KmerNBmodel_'+ str(k) + 'mer.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)




#some test data and helpers

"""
#count the distribution of the labels
import collections
counter = collections.Counter(genus_labels)
print(counter)
"""



"""
##sample a small subset of the data

from numpy.random import default_rng

size=1000
rng = default_rng()
indices = rng.choice(len(ss), size=size, replace=False)
ss_small=[ss[i] for i in indices]
genus_labels_small=genus_labels[indices]
"""


"""
#toy data

#ss=['atcg','gatc','tacg']
ss=['atcgatat','cgatcgcg','atcgcgat','gatcgaga','gatctctc','gatcgatc','tacgcgcg','tatatacg','cgtatacg']
y=[0,0,0,1,1,1,2,2,2]
k=4
"""

"""
#larger test data
import random
ss=[]
y=[]
for i in range(100):
    ss.append(''.join(random.choices('atcg',k=1000))) #generate a random sequence of length k
    y.append(random.choices([0,1,2],k=1)) #generate random labels
y=np.array(y)
"""