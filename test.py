#get subset of Silva for mock-2 prediction
from test_data import *
import numpy as np
from utility import readFasta, get_label_list, get_accu_f1, get_f1
from KmerConstruct import KmerConstruct
from KmerFeatureEng import KmerFeatureEng
from multinomialNB import multinomialNB
from dict_input_multinomialNB import dict_input_multinomialNB
import pandas as pd
import pickle


#construct X
k=4 #k=8 takes >10min for feature construction
ss_subset=[ss[i] for i in indices]
new_KmerConstruct=KmerFeatureEng(answers_,ss_subset,k=k)
new_KmerConstruct.constuct_feature_table()    
new_KmerConstruct.unpack_feature_table()
X=new_KmerConstruct.X 
y=y_mock2

#train
NB=multinomialNB(X_train=X,y_train=y,alpha=0.1,include_prior=True) #for current test include_prior=False doesn't make a difference
NB.train()


#predict
#construct X_test (k-mer frequency table) from mock data
KmerConstruct_mock2=KmerFeatureEng(answers_,seq_mock,k=k,n_threads=1)
KmerConstruct_mock2.features=new_KmerConstruct.features #map the k-mer features to those of the classifier's
KmerConstruct_mock2.constuct_feature_table()
KmerConstruct_mock2.unpack_feature_table()

X_test=KmerConstruct_mock2.X
y_test=answers_


y_pred=NB.test(X_test=X_test.A)
get_accu_f1(y_pred,y_test) 




#train dict_input_multinomialNB 
from dict_input_multinomialNB import dict_input_multinomialNB

k=4
new_KmerConstruct=KmerConstruct(ss_subset,k=k)
new_KmerConstruct.construct_all_freq_dict()
dict_NB=dict_input_multinomialNB(ls_dict_X_train=new_KmerConstruct.all_freq_dict,n_features=len(new_KmerConstruct.features),y_train=y,alpha=0.1,include_prior=True) 
dict_NB.train()


#construct X_test (k-mer frequency table) from mock data
KmerConstruct_mock=KmerConstruct(seq_mock,k=k,n_threads=1)
KmerConstruct_mock.construct_all_freq_dict()

y_pred=dict_NB.test(ls_dict_X_test=KmerConstruct_mock.all_freq_dict)
get_accu_f1(y_pred,y_test) #accu: 0.8, F1: 0.88



#use the most discriminative k to train the model


##use mini dataset
k=4
new_KmerConstruct=KmerFeatureEng(y_mini,ss_mini,k=k)

#dropping this to 0.9 yields almost no accuracy at all. why? 
#is the implementation of construct_discrit_freq_dict() even correct?
#maybe I can separate different discriminative kmer approaches in another class
new_KmerConstruct.construct_discrit_freq_dict(remain=.9) 

dict_NB=dict_input_multinomialNB(ls_dict_X_train=new_KmerConstruct.discrit_freq_dict,n_features=len(new_KmerConstruct.discrit_kmer_no_labels),y_train=y_mini,alpha=0.1,include_prior=True) 
dict_NB.train()        


y_pred=dict_NB.test(ls_dict_X_test=new_KmerConstruct.all_freq_dict) #list is correct but prediction for each sample is wrong given remain=0.5: [0, 0, 0, 1, 1, 0, 2, 0, 2]
get_accu_f1(y_pred,y_mini) 





##use larger dataset
k=32
new_KmerConstruct=KmerFeatureEng(y_mock2,ss_subset,k=k)

#dropping this to 0.9 yields almost no accuracy at all. why? 
#is the implementation of construct_discrit_freq_dict() even correct?
#maybe I can separate different discriminative kmer approaches in another class
new_KmerConstruct.construct_discrit_freq_dict(remain=1) 

dict_NB=dict_input_multinomialNB(ls_dict_X_train=new_KmerConstruct.discrit_freq_dict,n_features=len(new_KmerConstruct.discrit_kmer_no_labels),y_train=y_mock2,alpha=0.1,include_prior=True) 
dict_NB.train()        

KmerConstruct_mock=KmerConstruct(seq_mock,k=k,n_threads=1)
KmerConstruct_mock.construct_all_freq_dict()

y_pred=dict_NB.test(ls_dict_X_test=KmerConstruct_mock.all_freq_dict)
get_accu_f1(y_pred,y_test_mock2) #accu: 0.16, F1: 0.27586











#train dict_input_multinomialNB using complete silva data
new_KmerConstruct = pickle.load(open("../KmerConstruct_4mer.pickle","rb" ))
y=genus_labels


#train
dict_NB=dict_input_multinomialNB(ls_dict_X_train=new_KmerConstruct.all_freq_dict,n_features=len(new_KmerConstruct.features),y_train=y,alpha=0.1,include_prior=True) 
dict_NB.train() #15min 27s

#predict
#construct X_test (k-mer frequency table) from mock data
k=4
KmerConstruct_mock2=KmerConstruct(seq_mock2,k=k,n_threads=1)
KmerConstruct_mock2.construct_all_freq_dict()

y_test=answers_

y_pred=dict_NB.test(ls_dict_X_test=KmerConstruct_mock2.all_freq_dict) # 1 min 22 s
get_accu_f1(y_pred,y_test) #Accuracy=  0.2916666666666667 F1=  0.45161290322580644





from sklearn.naive_bayes import MultinomialNB #accu: 0.8, F1: 0.88
model = MultinomialNB(alpha=0.1) #alpha=5, 1, 0.1 make no difference
model.fit(X, y)
y_pred = model.predict(X_test.A)
get_accu_f1(y_pred,y_test)






#sklearn decision tree #accu: 0.8, F1: 0.88
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=0)
model.fit(X, y)
y_pred = model.predict(X_test.A) 
get_accu_f1(y_pred,y_test)


#sklearn gradient boosting  #accu: 0.76, F1: 0.86
from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,max_depth=1, random_state=0)
model.fit(X, y)
y_pred = model.predict(X_test.A) 
get_accu_f1(y_pred,y_test)


#sklearn random forest
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,max_depth=30, random_state=102)
model.fit(X, y)
y_pred = model.predict(X_test.A) 
get_accu_f1(y_pred,y_test)

#perfect classifier
#if there is no identical sequence with different labels, accu, f1 should both be 1 
for i,seq1 in enumerate(seq_mock2):
    for j,seq2 in enumerate(seq_mock2):
        if i!=j and seq1==seq2:
            print("seq",i," ",j,"is identical")
#=> True



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



#compare my multinomialNB with sklearn's using toy data

#toy data
from multinomialNB import multinomialNB

#raw data is loaded from test/test_data.py


#construct X
new_KmerConstruct=KmerFeatureEng(ss_mini)
new_KmerConstruct.constuct_feature_table()    
new_KmerConstruct.unpack_feature_table()
X=new_KmerConstruct.X  

NB=multinomialNB(X_train=X,y_train=y_mini)
NB.train()
y_pred=NB.test(X_test=X.A)
get_accu_f1(y_pred,y) #training error


from sklearn.naive_bayes import MultinomialNB #accu: 0.8, F1: 0.88
model = MultinomialNB(alpha=0.1) #alpha=5, 1, 0.1 make no difference
model.fit(X, y)
y_pred = model.predict(X.A)
get_accu_f1(y_pred,y)


#compare my multinomialNB with sklearn's using mid-sized toy data

#toy data
from multinomialNB import multinomialNB

#raw data is loaded from test/test_data.py


#construct X
new_KmerConstruct=KmerFeatureEng(ss_med,k=4)
new_KmerConstruct.constuct_feature_table()    
new_KmerConstruct.unpack_feature_table()
X=new_KmerConstruct.X  

NB=multinomialNB(X_train=X,y_train=y)
NB.train()
y_pred=NB.test(X_test=X.A)
get_accu_f1(y_pred,y) #training error


from sklearn.naive_bayes import MultinomialNB #accu: 0.8, F1: 0.88
model = MultinomialNB(alpha=0.1) #alpha=5, 1, 0.1 make no difference
model.fit(X, y)
y_pred = model.predict(X.A)
get_accu_f1(y_pred,y)






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