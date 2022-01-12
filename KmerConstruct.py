#basic kmer related functions

import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Process
import itertools  
import scipy.sparse as sparse


class KmerConstruct():
    
    def __init__(self, ss, k, n_threads=1,remove_non_standard_nt=True):
        self.ss = ss
        self.ss_kmers=None
        self.k = k
        self.X = None
        self.X_packed=None
        self.X_first_j=None
        self.n_threads=n_threads
        self.features=None
        self.remove_non_standard_nt=remove_non_standard_nt
        self.non_standard=['R','Y','K','M','S','W','B','D','H','V','N'] #http://www.hgmd.cf.ac.uk/docs/nuc_lett.html 
        self.all_freq_dict=None
        self.discrit_freq_dict=None

    def build_kmer_list(self,ss,k):
        
        #if k > max(seq length in ss), return 
        #get the len of the longest sequence in ss
        max_len=0
        for i in range(len(ss)):
            max_len=max(max_len,len(ss[i]))
            
        if k>max_len:
            raise ValueError("k is larger than the size of at least one input sequence!")
        
        print('==Building kmer lists==')
        ss_kmers=[]
        for s in tqdm(ss):
            s_kmer=[]
            for i in range(0,len(s)-k+1):
                kmer=s[i:i+k]
                keep=True
                if self.remove_non_standard_nt:
                    for nt in self.non_standard:
                        if kmer.find(nt)!=-1:
                            keep=False
                            break
                    if keep: s_kmer.append(kmer)
            
            ss_kmers.append(s_kmer)
        return ss_kmers
    
    def count_freq(self,kmer_list):
        freq_dict={}
        for kmer in kmer_list:
            if kmer not in freq_dict:
                freq_dict[kmer]=1
            else:
                freq_dict[kmer]+=1
        return(freq_dict)
    
    def count_all_freq(self,ss_kmers):
        freq_dic_list=[]
        for kmers in tqdm(ss_kmers):
            freq_dic_list.append(self.count_freq(kmers))
        return freq_dic_list
    
    def get_all_uniq_kmers(self,ss_kmers):
        #flatten the kmers: list of lists -> list
        print('==Getting all unique kmers==')
        flattened_ss_kmers=[s_kmer for s_kmers in tqdm(ss_kmers) for s_kmer in s_kmers]
        return sorted(list(set(flattened_ss_kmers))) #it has to be sorted. Otherwise set gives different orders at each run
    
    
    def convert_kmer_into_index(self,batch_freq_dict):
        batch_freq_tuple=[]
        for freq_dict in tqdm(batch_freq_dict):
            freq_tuple=[]
            for k, v in freq_dict.items():
                ind=self.features.index(k)
                freq_tuple.append((ind, v))
            batch_freq_tuple.append(freq_tuple)
        return batch_freq_tuple
    
    
    def fill_feature(self,batch_freq_tuple):
        X = sparse.lil_matrix((len(batch_freq_tuple),len(self.features)), dtype=int)
        for i,row in enumerate(tqdm(batch_freq_tuple)):
            for col in row:
                 X[i, col[0]] = col[1]
        return X
    
    
    
    def construct_all_freq_dict(self):
        print('getting kmer frequencies')
       
        self.ss_kmers=self.build_kmer_list(self.ss,self.k)
        if self.features is None:
            self.features=self.get_all_uniq_kmers(self.ss_kmers)
        else:
            print('==kmer features are already given==')
        
        if self.remove_non_standard_nt:
            new_features=[]
            for i in range(len(self.features)):
                keep=True
                for nt in self.non_standard:
                    if self.features[i].find(nt)!=-1:
                        keep=False
                        break
                
                if keep: new_features.append(self.features[i])
            
            self.features=new_features 
        
        self.all_freq_dict=self.count_all_freq(self.ss_kmers) #there's a tqdm within this function
        
    
    def constuct_feature_table(self): #for different samples, merge their k-mer frequencies in one feature table. j: get the most discriminative kmers (features)
           
        if self.X is None:                                  
            print('==constructing the full feature table==')
            
            if self.all_freq_dict is None:
                self.construct_all_freq_dict()
            
            print('converting kmers into feature indices')
            self.all_freq_tuple=[]
            
            if self.n_threads==1:           
                self.all_freq_tuple=self.convert_kmer_into_index(batch_freq_dict=self.all_freq_dict) #there's a tqdm within this function
            else:                
                #split self.all_freq_dict into chunks of size n (last chunk<=n)
                batch_freq_dict_list=[]
                n=len(self.all_freq_dict)//self.n_threads
                for i in range(0,len(self.all_freq_dict),len(self.all_freq_dict)//self.n_threads):
                    batch_freq_dict_list.append(self.all_freq_dict[i:i+n])

                pool = Pool(self.n_threads)
                results=pool.map(self.convert_kmer_into_index,batch_freq_dict_list)
                for result in results:  
                    self.all_freq_tuple.extend(result)
            
        

            
                
