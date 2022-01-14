#test takes ~8hrs to finish

import pickle
from utility import readFasta, get_label_list, get_accu_f1
from KmerConstruct import KmerConstruct
import pandas as pd
from os.path import exists 

no_dataset=set([11,13,14,15,17,25]) #datasets skipped in tax-credit
no_expected_tsv=set([1,2,6,7,8]) # no expected-sequence-taxonomies.tsv -> read matched-sequence-taxonomies.tsv instead
ITS=set([9,10,24,26])
ls=[]
for i in range(1,26+1):
    if i not in (no_dataset.union(ITS)):        
        ls.append('mock-'+str(i))

#construct test data
all_mock_accu_f1={}
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
        
        #map the features to those of the classifiers'
        ks=[4,6,7,8,9,10,11,12,14,16,18,32,64,100]
        
        mock_accu_f1=[]
        for k in ks:
            filename='../models/515F_806R_all_3_domains/NBmodel_'+ str(k) + 'mer.pickle'
            if exists(filename):        
                model = pickle.load(open(filename,"rb" ))
                KmerConstruct_mock=KmerConstruct(seq_mock,k=k,n_threads=1)
                
                try: #rep seqs for some mocks (eg. mock6) are less than k 
                    KmerConstruct_mock.construct_all_freq_dict()
                except:
                    mock_accu_f1.append(['k='+ str(k) +' is larger than rep seqs'])
                    all_mock_accu_f1[mock]=mock_accu_f1
                    continue
                    
                y_pred = model.test(KmerConstruct_mock.all_freq_dict) 
                accu,f1=get_accu_f1(y_pred,y_test)
                mock_accu_f1.append([accu,f1])
        two_test_mock_accu_f1[seq_file]=mock_accu_f1
    all_mock_accu_f1[mock]=two_test_mock_accu_f1
                
                #print("Accuracy= ",accu," for k=",k) #grace won't print this
                #print("F1= ",f1," for k=",k) #grace won't print this



#save the numbers of evaluation 
with open('../results/all_mock_accu_f1.pickle','wb') as handle:    
    pickle.dump(all_mock_accu_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

#plot
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


        
    

    
    

    









