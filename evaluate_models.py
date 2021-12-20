import pickle
from utility import readFasta, get_label_list, get_accu_f1
from KmerConstruct import KmerConstruct
import pandas as pd


#pick a mock and do prediction on eg. genus
new_KmerConstruct = pickle.load(open("../KmerConstruct_4mer.pickle","rb" ))
#model= pickle.load(open("../KmerNBmodel_4mer.pickle","rb" ))
#model=

#construct test_X (k-mer frequency table) from mock daxxwtax
label_mock2,seq_mock2=readFasta('../tax-credit-data/data/mock-community/mock-2/dna-sequences.fasta') 
taxonomy_table=pd.read_csv('../tax-credit-data/data/mock-community/mock-2/matched-sequence-taxonomies.tsv',sep='\t')
labels=taxonomy_table['Standard Taxonomy'].tolist()
answers_=get_label_list(labels,type_='g__')

KmerConstruct_mock2=KmerConstruct(seq_mock2,k=4,n_threads=1)
KmerConstruct_mock2.features=new_KmerConstruct.features #map the k-mer features to those of the classifier
KmerConstruct_mock2.constuct_feature_table()

X_test=KmerConstruct_mock2.X
#map the features to those of the classifiers'


y=answers_
y_pred = model.predict(X_test) 
get_accu_f1(y_pred,y)



#plot
