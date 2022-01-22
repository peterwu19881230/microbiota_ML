#get the k-mer frequencies for the given sequences
#aim: get the most discriminative k-mers, train an multinomial NB classifier using sklearn:


import os
from os.path import exists
#os.chdir('/Users/ifanwu/Documents/CS_MS_TAMU/microbiota_ML/github') #macos only

from utility import readFasta, get_label_list, get_accu_f1
from KmerFeatureEng import KmerFeatureEng

import multiprocessing
from multiprocessing import Pool, Process
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

import pickle


if __name__ == '__main__':
    with open('load_silva_data.py') as infile:
        exec(infile.read())
    
    out_dir='../kmer_feature_objects/'
    
    #construct kmer feature table 
    for k in [4]: #[32,64,100]
        for remain in [0.95,0.9,0.8,0.5,0.25,0.1]:
            filename= out_dir+ 'KmerConstruct_'+ str(k) + 'mer_'+ str(remain)+ 'remained_.pickle'
            if not exists(filename):        
                #construct kmer feature table
                new_KmerConstruct=KmerFeatureEng(genus_labels,ss,k,n_threads=1)
                new_KmerConstruct.construct_discrit_freq_dict()
                print('saving the file...')
                with open(filename,'wb') as handle:
                    pickle.dump(new_KmerConstruct, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('kmer freq for k=',k,' remain=',remain ,' is done')
        

