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
from multiprocessing import Pool




def constuct_(k):
	
	#can't just do the following. Don't know why
	"""
	with open('load_silva_data.py') as infile:
		exec(infile.read())
	"""
	
	from utility import readFasta, get_label_list

	#load data from silva
	#there are sequences lost after truncation. Retrieve the id indices and applied that on the taxonomy labels
	label_silva_138_99_V4_truncated,seq_silva_138_99_V4_truncated=readFasta('../silva/silva138_99_truncated.fasta') 
	label_silva_138_99_V4,seq_silva_138_99_V4=readFasta('../silva/silva138_99.fasta') 

	indices_after_truncation=[]
	curr_index=0
	for i in range(len(label_silva_138_99_V4_truncated)):
		while label_silva_138_99_V4_truncated[i]!=label_silva_138_99_V4[curr_index]:
			curr_index+=1
		indices_after_truncation.append(curr_index)    
			
	
	ss=seq_silva_138_99_V4_truncated


	#read taxonomy data
	import pandas as pd
	taxonomy_table=pd.read_csv('../silva/taxonomy.tsv',sep='\t')
	labels=taxonomy_table['Taxon'].tolist()


	#get genus as labels
	genus_labels=get_label_list(labels,type_='g__')
	genus_labels=np.array(genus_labels)
	genus_labels=genus_labels[indices_after_truncation]
	
	#construct kmer feature table 
	filename='../kmer_feature_objects/KmerFeatureEng_'+ str(k) + 'mer.pickle'
	if not exists(filename):        
		#construct kmer feature table
		new_KmerFeatureEng=KmerFeatureEng(genus_labels,ss,k,n_threads=1)
		new_KmerFeatureEng.construct_all_freq_dict()
		print('saving the file...')
		with open(filename,'wb') as handle:
			pickle.dump(new_KmerFeatureEng, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print('kmer freq for k=',k,' is done')


if __name__ == '__main__':
	ks=[4,6,7,8,9,10,11,12,14,16,18,32,64,100]
	n_threads=len(ks)
	pool = Pool(n_threads)
	pool.map(constuct_,ks)


	

