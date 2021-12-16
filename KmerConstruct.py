#this class contains kmer related functions

import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Process


class KmerConstruct():
    
    def __init__(self, ss, k, n_threads=1):
        self.ss = ss
        self.k = k
        self.X = None
        self.X_first_j=None
        self.n_threads=n_threads

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
                s_kmer.append(kmer)
            
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
    
    def get_all_uniq_kmers(self,ss_kmers):
        #flatten the kmers: list of lists -> list
        print('==Getting all unique kmers==')
        flattened_ss_kmers=[s_kmer for s_kmers in tqdm(ss_kmers) for s_kmer in s_kmers]
        return list(set(flattened_ss_kmers))
    
    def fill_feature(self,batch):
        for i in tqdm(batch):
            for j in range(self.X.shape[1]):   
                self.X[i,j]=self.ss_kmers[i].count(self.features[j])
    
    
    def constuct_feature_table(self,j_=None): #for different samples, merge their k-mer frequencies in one feature table. j: get the most discriminative kmers (features)
           
        if self.X is None:
            self.ss_kmers=self.build_kmer_list(self.ss,self.k)
            self.features=self.get_all_uniq_kmers(self.ss_kmers)
            self.X=np.zeros((len(self.ss_kmers),len(self.features)),dtype='int32')
            
            print('==constructing the full feature table==')
            processes=[]
            batches=[]
            for i in range(self.n_threads): 
            	batch=range(int(self.X.shape[0]/self.n_threads)*i,int(self.X.shape[0]/self.n_threads)*(i+1))
            	batches.append(batch)

            for batch in batches:
                print('A thread has started..')
                if self.n_threads==1:
                    self.fill_feature(batch) #non-paralle version
                else:
                    p=Process(target=self.fill_feature,args=(batch,))             
                    p.start()
            
        
                    
        
        if j_ is not None:
            #calculate no. of 0s for each column and sort by desc order
            X_t=self.X.T
            sum_zeros=[]
            print('==Getting the most discriminative features==')
            col_len=len(self.ss_kmers)
            for column in tqdm(X_t):  
                sum_zeros.append(col_len-np.count_nonzero(column))
                """
                In latter versions I probably have to think about situations where all cells in the feature table don't contain any 0s
                """
            
            new_X_t=[]
            while j_ != 0:
                col_index=sum_zeros.index(max(sum_zeros))
                sum_zeros.pop(col_index)
                new_X_t.append(X_t[col_index])
                j_-=1
            new_X_t=np.array(new_X_t)    
            self.X_first_j=new_X_t.T
            
                
