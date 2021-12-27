#get subset of Silva for mock-2 prediction
import numpy as np
from utility import readFasta, get_label_list, get_accu_f1
from KmerConstruct import KmerConstruct
from multinomialNB import multinomialNB
import pandas as pd
import pickle

#seqs, labels from mock-2
label_mock2,seq_mock2=readFasta('../tax-credit-data/data/mock-community/mock-2/dna-sequences.fasta') 
taxonomy_table=pd.read_csv('../tax-credit-data/data/mock-community/mock-2/matched-sequence-taxonomies.tsv',sep='\t')
labels=taxonomy_table['Standard Taxonomy'].tolist()
answers_=get_label_list(labels,type_='g__')

#pick sequences, labels that match mock-2 from Silva
with open('load_silva_data.py') as infile:
    exec(infile.read()) #ss, genus_labels are given here
    
indices=[]
for i,label in enumerate(genus_labels): 
 if label in answers_: 
     indices.append(i)

y=genus_labels[indices]

#construct X
k=4 #k=8 takes >10min for feature construction
ss_subset=[ss[i] for i in indices]
new_KmerConstruct=KmerConstruct(ss_subset,k=k)
new_KmerConstruct.constuct_feature_table()    
new_KmerConstruct.unpack_feature_table()
X=new_KmerConstruct.X 


#train
NB=multinomialNB(X_train=X,y_train=y,alpha=0.01)
NB.train()


#predict
#construct X_test (k-mer frequency table) from mock data
KmerConstruct_mock2=KmerConstruct(seq_mock2,k=k,n_threads=1)
KmerConstruct_mock2.features=new_KmerConstruct.features #map the k-mer features to those of the classifier's
KmerConstruct_mock2.constuct_feature_table()
KmerConstruct_mock2.unpack_feature_table()

X_test=KmerConstruct_mock2.X
y_test=answers_


y_pred=NB.test(X_test=X_test.A)
get_accu_f1(y_pred,y_test) 


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=0.1) #alpha=5, 1, 0.1 make no difference
model.fit(X, y)
y_pred = model.predict(X_test.A)
get_accu_f1(y_pred,y_test)

#with open('../KmerNBmodel_'+ str(k) + 'mer.pickle', 'wb') as handle:
#    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)



#sklearn decision tree (currently the best model: accu: 0.95, F1: 0.97)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=0)
model.fit(X, y)
y_pred = model.predict(X_test.A) 
get_accu_f1(y_pred,y)


#sklearn gradient boosting (accu: 0.90, F1: 0.95))
from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,max_depth=1, random_state=0)
model.fit(X, y)
y_pred = model.predict(X_test.A) 
get_accu_f1(y_pred,y)


#perfect classifier




#build my NB on full data (not done)
"""
import numpy as np
from utility import readFasta, get_label_list, get_accu_f1
from KmerConstruct import KmerConstruct
from multinomialNB import multinomialNB
import pandas as pd


with open('load_silva_data.py') as infile:
    exec(infile.read())
    
k=4
y=genus_labels
new_KmerConstruct = pickle.load(open("../KmerConstruct_4mer.pickle","rb" ))
X=new_KmerConstruct.X  


#train
NB=multinomialNB(X[0:1000],y[0:1000])
NB.train()



#predict using mock2

#construct X_test (k-mer frequency table) from mock data
label_mock2,seq_mock2=readFasta('../tax-credit-data/data/mock-community/mock-2/dna-sequences.fasta') 
taxonomy_table=pd.read_csv('../tax-credit-data/data/mock-community/mock-2/matched-sequence-taxonomies.tsv',sep='\t')
labels=taxonomy_table['Standard Taxonomy'].tolist()
answers_=get_label_list(labels,type_='g__')

KmerConstruct_mock2=KmerConstruct(seq_mock2,k=4,n_threads=1)
KmerConstruct_mock2.features=new_KmerConstruct.features #map the k-mer features to those of the classifier's
KmerConstruct_mock2.constuct_feature_table()
KmerConstruct_mock2.unpack_feature_table()

X_test=KmerConstruct_mock2.X.A
y_test=answers_


y_pred=NB.test(X_test,y_test)

get_accu_f1(y_pred,y)
"""



#toy data
"""
#raw data
ss=['atcgatat','cgatcgcg','atcgcgat','gatcgaga','gatctctc','gatcgatc','tacgcgcg','tatatacg','cgtatacg']
y=[0,0,0,1,1,1,2,2,2]
k=4

#construct X
new_KmerConstruct=KmerConstruct(ss,k)
new_KmerConstruct.constuct_feature_table()    
new_KmerConstruct.unpack_feature_table()
X=new_KmerConstruct.X  

NB=multinomialNB(X_train=X,y_train=y)
NB.train()
y_pred=NB.test(X_test=X.A)
get_accu_f1(y_pred,y) #training error
"""




#test if my code handles memory
"""
import sys
import numpy as np
from KmerConstruct import KmerConstruct
with open('load_silva_data.py') as infile:
    exec(infile.read())
new_KmerConstruct=KmerConstruct(ss,k=4)
new_KmerConstruct.constuct_feature_table()    
new_KmerConstruct.unpack_feature_table()
X=new_KmerConstruct.X    
sys.getsizeof(X)
"""


""" #calculate # features for different k
import numpy as np
from KmerConstruct import KmerConstruct
with open('load_silva_data.py') as infile:
    exec(infile.read())
        
new_KmerConstruct=KmerConstruct(ss,k=64)
ss_kmers=new_KmerConstruct.build_kmer_list(ss,k=64)
all_kmer_list=new_KmerConstruct.count_all_freq(ss_kmers)
all_uniq_kmers=new_KmerConstruct.get_all_uniq_kmers(ss_kmers)
len(all_uniq_kmers)


#for silva and k=8, there are 65257 unique kmers, about the same as than 4**8=65536 kmers
#for silva and k=16, there are 2498160 unique kmers, much less than 4**16 kmers
#for silva and k=32, there are 6236289 unique kmers, much less than 4**32 kmers
#for silva and k=64, there are 7864011 unique kmers, much less than 4**64 kmers
#for silva and k=100, there are 3586555 unique kmers, much less than 4**100 kmers
"""

"""
#how do I use this c++ plugin? (Or do I need to)
import jellyfish
"""

#test sparse matrix
"""
#https://stackoverflow.com/questions/43618956/constructing-sparse-matrix-from-list-of-lists-of-tuples
import numpy as np
import scipy.sparse as sparse
import sys
import itertools
from tqdm import tqdm


#alist = [[(1,10), (3,-3)],[(2,12)]]
alist = []
for kmer_dict in tqdm(all_kmer_list[0:5]): #too slow for even 1 iteration
    list_=[]
    for key in kmer_dict:
        list_.append((all_uniq_kmers.index(key),kmer_dict[key]))
    alist.append(list_)


M = sparse.lil_matrix((400000,7864011), dtype=int)
for i,row in enumerate(alist):
     for col in row:
         M[i, col[0]] = col[1]
M.A
sys.getsizeof(M)
"""



"""
#sparse matrix
import numpy as np
import scipy.sparse as sparse
import pickle

new_KmerConstruct = pickle.load(open("../KmerConstruct_4mer.pickle","rb" ))
X_np=new_KmerConstruct.X
X_csr=sparse.csr_matrix(new_KmerConstruct.X)
X_lil=sparse.lil_matrix(new_KmerConstruct.X)

with open('../X_np', 'wb') as handle:
    pickle.dump(X_np, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../X_csr', 'wb') as handle:
    pickle.dump(X_csr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../X_lil', 'wb') as handle:
    pickle.dump(X_lil, handle, protocol=pickle.HIGHEST_PROTOCOL)

features=new_KmerConstruct.features
ss_kmers=new_KmerConstruct.ss_kmers
"""


#test toy data
"""
from KmerConstruct import KmerConstruct
ss=['ATCTATCG','TCGAGATC','TAGCTGAC','TAGGGGGG','TACACATT','ATGATCAA']
k=3

new_KmerConstruct=KmerConstruct(ss,k,n_threads=1) #n_threads>=2 won't work in spyder
new_KmerConstruct.constuct_feature_table()
new_KmerConstruct.unpack_feature_table()
""" 



#test part of silva data
"""
import numpy as np
from KmerConstruct import KmerConstruct
with open('load_silva_data.py') as infile:
    exec(infile.read())
new_KmerConstruct=KmerConstruct(ss[0:10],k=4)
new_KmerConstruct.constuct_feature_table()    
new_KmerConstruct.unpack_feature_table()    
new_KmerConstruct.all_freq_tuple
"""

"""
def convert_seq_to_ind(seq):
    dict_={'A':0,'C':1,'G':2,'T':3}
    ind=0
    i=0
    for nt in reversed(seq):
        ind+=dict_[nt]*4**i
        i+=1
        
    return ind
"""


##random subset of Silva
"""

from numpy.random import default_rng

size=1000
rng = default_rng()
indices = rng.choice(len(ss), size=size, replace=False)
ss_small=[ss[i] for i in indices]
genus_labels_small=genus_labels[indices]
"""


#random data (test if the program runs)
"""
import random
ss=[]
y=[]
for i in range(100):
    ss.append(''.join(random.choices('atcg',k=1000))) #generate a random sequence of length k
    y.append(random.choices([0,1,2],k=1)) #generate random labels
y=np.array(y)
"""

#some test data and helpers

"""
#count the distribution of the labels
import collections
counter = collections.Counter(genus_labels)
print(counter)
"""