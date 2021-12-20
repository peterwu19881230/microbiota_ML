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


if __name__ == '__main__':
	execfile('load_silva_data.py')
    ks=[4,6,7,8,9,10,11,12,14,16,18,32,64,100]
    n_threads=20
    #construct kmer feature table 
    for k in ks:
        #construct kmer feature table
        new_KmerConstruct=KmerConstruct(ss,k,n_threads=n_threads)
        new_KmerConstruct.constuct_feature_table()
        with open('../KmerConstruct_'+ str(k) + 'mer.pickle', 'wb') as handle:
            pickle.dump(new_KmerConstruct, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

