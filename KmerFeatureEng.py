#kmer feature engineering

import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Process
import itertools  
import scipy.sparse as sparse
from KmerConstruct import KmerConstruct


class KmerFeatureEng(KmerConstruct):
    
    def __init__(self, labels, ss, k, n_threads=1,remove_non_standard_nt=True):
        self.labels=labels
        super().__init__(ss, k, n_threads=1,remove_non_standard_nt=True)

        
    #kmers present in a class but not in others
    def construct_discrit_freq_dict(self,remain=0.5):
        if self.all_freq_dict is None:
            self.construct_all_freq_dict()
            
        #get no. of labels(y) each kmer is present for the whole data and sort
        print('==counting kmer presence in each label(y)==')
        
        ##dict of kmer dict for each label
        uniq_labels=sorted(list(set(self.labels)))
        kmer_for_each_label={}
        for label in tqdm(uniq_labels):
            indices=np.where(self.labels==label)[0].tolist()
            label_freq_dict=[self.all_freq_dict[i] for i in indices]
            sum_label_freq_dict={}
            for dict_ in label_freq_dict:
                for kmer in dict_:                        
                    if kmer not in sum_label_freq_dict:
                        sum_label_freq_dict[kmer]=dict_[kmer]
                    else:
                        sum_label_freq_dict[kmer]+=dict_[kmer]
            
            kmer_for_each_label[label]=sum_label_freq_dict
       
        ##dict of frequencies for each label for each kmer
        feature_counts_each_label={}
        for feature in self.features:
           for k in kmer_for_each_label:
               if feature in kmer_for_each_label[k]:
                   if feature not in feature_counts_each_label:
                       feature_counts_each_label[feature]=1
                   else:
                       feature_counts_each_label[feature]+=1
                       
        feature_counts_each_label={k: v for k, v in sorted(feature_counts_each_label.items(), key=lambda item: item[1])} 
        
        print('==constructing discriminative kmer features==')
        n_remained=round(len(self.features)*remain)
        self.discrit_kmer_no_labels={A:N for (A,N) in [x for x in feature_counts_each_label.items()][:n_remained]}  
        
        self.discrit_ss_kmers=[]
        for i in tqdm(range(len(self.ss_kmers))):
            ss_kmer=[]
            for j in range(len(self.ss_kmers[i])):
                if self.ss_kmers[i][j] in self.discrit_kmer_no_labels:
                    ss_kmer.append(self.ss_kmers[i][j])

            self.discrit_ss_kmers.append(ss_kmer) 
            
        self.discrit_freq_dict=self.count_all_freq(self.discrit_ss_kmers) #there's a tqdm within this function 
        
        
        
    def unpack_feature_table(self,j_=None): #unpacking on my mac/pc will crash
        print('==Unpacking the feature table==')
        
        if self.n_threads==1:
            self.X=self.fill_feature(self.all_freq_tuple)
            
        else:
            batch_freq_tuple_list=[]
            n=len(self.all_freq_tuple)//self.n_threads
            for i in range(0,len(self.all_freq_tuple),len(self.all_freq_tuple)//self.n_threads):
                batch_freq_tuple_list.append(self.all_freq_tuple[i:i+n])
            pool = Pool(self.n_threads)
            results=pool.map(self.fill_feature,batch_freq_tuple_list)
            self.X=sparse.vstack(tuple(results))

        
