from utility import readFasta, get_label_list
import pandas as pd
import numpy as np

#synthetic data
ss_mini=['atcgatat','cgatcgcg','atcgcgat','gatcgaga','gatctctc','gatcgatc','tacgcgcg','tatatacg','cgtatacg']
y_mini=[0,0,0,1,1,1,2,2,2]

np.random.seed(101)
ss_med=[''.join(np.random.choice(['a','t','c','g'],30)) for _ in range(100)]
y_med=[i for i in range(10) for _ in range(10)]




#subset of Silva that has mock-2 labels
mock='mock-2'
_,seq_mock=readFasta('../tax-credit-data/data/mock-community/'+ mock+ '/dna-sequences.fasta') 

#taxonomy_table=pd.read_csv('../tax-credit-data/data/mock-community/'+ mock+ '/expected-sequence-taxonomies.tsv',sep='\t')
taxonomy_table=pd.read_csv('../tax-credit-data/data/mock-community/'+ mock+ '/matched-sequence-taxonomies.tsv',sep='\t')
labels=taxonomy_table['Standard Taxonomy'].tolist()
answers_=get_label_list(labels,type_='g__')
y_test_mock2=answers_

#pick sequences, labels that match mock-2 from Silva
with open('load_silva_data.py') as infile:
    exec(infile.read()) #ss, genus_labels are given here
 
 
    
indices=[]
for i,label in enumerate(genus_labels): 
 if label in answers_: 
     indices.append(i)

ss_subset=[ss[i] for i in indices]
y_mock2=np.array(genus_labels)[indices]
y_mock2=y_mock2.tolist()



