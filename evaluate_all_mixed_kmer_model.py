#Making this to evaluate: ../models/NBmodel_mix_kmer.pickle


#test takes ~8hrs to finish

import pickle
from utility import readFasta, get_label_list, get_accu_f1
from KmerConstruct import KmerConstruct
import numpy as np
import pandas as pd
from os.path import exists 
from multiprocessing import Pool


from tqdm import tqdm

def combine_diff_kmer_features(KmerConstruct_list):
    mixed_feature_lists=[] 
    no_samples=len(KmerConstruct_list[0].all_freq_dict)
    for i in tqdm(range(no_samples)): #tqdm doesn't show up here. Don't know why
        mixed_feature_dict={}
        for j in range(len(KmerConstruct_list)):
            kmer_feature_dict=KmerConstruct_list[j].all_freq_dict[i]
            mixed_feature_dict={**mixed_feature_dict,**kmer_feature_dict}
                
        mixed_feature_lists.append(mixed_feature_dict)
            
    return mixed_feature_lists




def evaluate(ls):
    #construct test data
    for mock in ls: #diff between dna-sequences and expected-sequences?
        print('--------------------- ')
        print('Evaluating '+mock)
        print('--------------------- ')
        
        two_test_mock_accu_f1={}
        for seq_file in ['dna-sequences.fasta','expected-sequences.fasta']:
            _,seq_mock=readFasta('../tax-credit-data/data/mock-community/'+mock+'/'+seq_file) 
            
            
            if mock in no_expected_tsv:
            	file = '/expected-sequence-taxonomies.tsv'
            else:
                file = '/matched-sequence-taxonomies.tsv'
                
            taxonomy_table=pd.read_csv('../tax-credit-data/data/mock-community/'+mock+'/'+file,sep='\t')
            labels=taxonomy_table['Standard Taxonomy'].tolist()
            answers_=get_label_list(labels,type_='g__')
            
            
            y_test=answers_
            
    
            
            mock_accu_f1=[]
            KmerConstruct_list=[]
            
            for k in ks:
                KmerConstruct_mock=KmerConstruct(seq_mock,k=k,n_threads=1)
                
                try: #rep seqs for some mocks (eg. mock6) are less than k 
                    KmerConstruct_mock.construct_all_freq_dict()
                except:
                    print('k='+ str(k) +' is larger than rep seqs')
                    continue
                
                KmerConstruct_list.append(KmerConstruct_mock)
                
            mixed_feature_lists=combine_diff_kmer_features(KmerConstruct_list)    
                
                
            y_pred = model.test(mixed_feature_lists) 
            accu,f1=get_accu_f1(y_pred,y_test)
            mock_accu_f1.append([accu,f1])
            two_test_mock_accu_f1[seq_file]=mock_accu_f1
    
    return two_test_mock_accu_f1


if __name__ == '__main__':
    
    no_dataset=set([11,13,14,15,17,25]) #datasets skipped in tax-credit
    no_expected_tsv=set([1,2,6,7,8]) # no expected-sequence-taxonomies.tsv -> read matched-sequence-taxonomies.tsv instead
    ITS=set([9,10,24,26])
    ls=[]
    for i in range(1,26+1):
        if i not in (no_dataset.union(ITS)):        
            ls.append('mock-'+str(i))

    #ls=['mock-1','mock-2'] #for test
    ks=[4,6,7,8,9,10,11,12,14,16,18,32,64,100]
    filename='../models/NBmodel_mix_kmer.pickle'
    model = pickle.load(open(filename,"rb" ))
    

    n_threads=len(ls)
    pool = Pool(n_threads)
    all_mock_accu_f1_list=pool.map(evaluate,ks)
    
    all_mock_accu_f1={}
    for i,mock in enumerate(ls):
        all_mock_accu_f1[mock]=all_mock_accu_f1_list[i]
    
    
    #save the numbers of evaluation 
    with open('../results/all_mock_mixed_all_kmers_accu_f1.pickle','wb') as handle:    
        pickle.dump(all_mock_accu_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    all_mock_accu_f1=pickle.load(open('../results/all_mock_mixed_all_kmers_accu_f1.pickle','rb'))
        


#Have to finish changing the following code...

#plot
"""
import matplotlib.pyplot as plt

for seq_file in ['dna-sequences.fasta','expected-sequences.fasta']:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('+'.join(ls)+' - \n'+seq_file)
    fig.tight_layout()
    
    all_genus_accus_f1s={}
    for mock in ls:
        accus=[]
        f1s=[]       
        for i,k in enumerate(ks):     
            
            if len(all_mock_accu_f1[mock][seq_file][i])==1:#if k is larger than the rep seqs
                accus.append(0)
                f1s.append(0)
                continue
            
            accus.append(all_mock_accu_f1[mock][seq_file][i][0])
            f1s.append(all_mock_accu_f1[mock][seq_file][i][1])

        x=ks
        y1=accus
        y2=f1s
        
        ax1.plot(x, y1,'o-',alpha=0.4)
        ax1.set_ylim([0, 1.1])
        plt.setp(ax1, ylabel='Accuracy')
        
        ax2.plot(x, y2,'o-',alpha=0.4)
        ax2.set_ylim([0, 1.1])
        ax2.legend((ls),prop={'size': 8},framealpha=1,bbox_to_anchor=(1.5, 1), loc='upper right', ncol=1)
        plt.setp(ax2, ylabel='F1')


#get mean accu, f1 for all mocks
accu_f1_all_k_dict={}
for i,k in enumerate(ks):
    this_kmer_accus=[[],[]] #1st: dna-sequences.fasta, 2nd: expected-sequences.fasta  
    this_kmer_f1s=[[],[]]    
    
    accu_f1_all_k_dict[k]={'dna-sequences.fasta':[],'expected-sequences.fasta':[]}
    for mock in all_mock_accu_f1:    
        accu_f1=all_mock_accu_f1[mock]['dna-sequences.fasta'][i]
        
        if len(accu_f1)==2: 
            this_kmer_accus[0].append(accu_f1[0])
            this_kmer_f1s[0].append(accu_f1[1])
        else: #k is larger than rep seqs
            this_kmer_accus[0].append(0)
            this_kmer_f1s[0].append(0)

        accu_f1=all_mock_accu_f1[mock]['expected-sequences.fasta'][i]
        this_kmer_accus[1].append(accu_f1[0])
        this_kmer_f1s[1].append(accu_f1[1])
        
    accu_f1_all_k_dict[k]['dna-sequences.fasta']=[np.mean(this_kmer_accus[0]),np.mean(this_kmer_f1s[0])]
    accu_f1_all_k_dict[k]['expected-sequences.fasta']=[np.mean(this_kmer_accus[1]),np.mean(this_kmer_f1s[1])]
"""    
    

        









