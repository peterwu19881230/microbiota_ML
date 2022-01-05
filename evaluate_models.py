import pickle
from utility import readFasta, get_label_list, get_accu_f1
from KmerConstruct import KmerConstruct
import pandas as pd
from os.path import exists 


#pick a mock and do prediction on eg. genus
new_KmerConstruct = pickle.load(open("../KmerConstruct_4mer.pickle","rb" ))

#construct test_X (k-mer frequency table) from mock daxxwtax
label_mock2,seq_mock2=readFasta('../tax-credit-data/data/mock-community/mock-2/expected-sequences.fasta') 
taxonomy_table=pd.read_csv('../tax-credit-data/data/mock-community/mock-2/matched-sequence-taxonomies.tsv',sep='\t')
labels=taxonomy_table['Standard Taxonomy'].tolist()
answers_=get_label_list(labels,type_='g__')


y_test=answers_

#map the features to those of the classifiers'
ks=[4,6,7,8,9,10,11,12,14,16,18,32,64,100]

for k in ks:
    filename='../models/NBmodel_'+ str(k) + 'mer.pickle'
    if exists(filename):        
        model = pickle.load(open(filename,"rb" ))
        KmerConstruct_mock2=KmerConstruct(seq_mock2,k=k,n_threads=1)
        KmerConstruct_mock2.construct_all_freq_dict()
        y_pred = model.test(KmerConstruct_mock2.all_freq_dict) 
        get_accu_f1(y_pred,y_test)


#plot

#when k=10: Accuracy=  0.3953488372093023 F1=  0.5666666666666667
#when k=16: Accuracy=  0.43902439024390244 F1=  0.6101694915254238
