#get the k-mer frequencies for the given sequences
#aim: get the most discriminative k-mers, train an multinomial NB classifier using sklearn:


import os
from os.path import exists
#os.chdir('/Users/ifanwu/Documents/CS_MS_TAMU/microbiota_ML/github') #macos only

from utility import readFasta, get_label_list, get_accu_f1
from KmerFeatureEng import KmerFeatureEng

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

import pickle


if __name__ == '__main__':
    with open('load_silva_data.py') as infile:
        exec(infile.read())

    ks=[4,6,7,8,9,10,11,12,14,16,18,32,64,100]
    
    #construct kmer feature table 
    for k in ks:
        filename='../kmer_feature_objects/KmerFeatureEng_'+ str(k) + 'mer.pickle'
        if not exists(filename):        
            #construct kmer feature table
            new_KmerFeatureEng=KmerFeatureEng(genus_labels,ss,k,n_threads=1)
            new_KmerFeatureEng.construct_all_freq_dict()
            print('saving the file...')
            with open(filename,'wb') as handle:
                pickle.dump(new_KmerFeatureEng, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('kmer freq for k=',k,' is done')
        

