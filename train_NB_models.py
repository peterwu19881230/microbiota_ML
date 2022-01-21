from dict_input_multinomialNB import dict_input_multinomialNB
import pickle
from os.path import exists 
import numpy as np
from multiprocessing import Pool


def train(k):
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
	
	filename='../kmer_feature_objects/KmerFeatureEng_'+ str(k) + 'mer.pickle'
	if exists(filename):        
		new_KmerConstruct = pickle.load(open(filename,"rb" ))
		
		model=dict_input_multinomialNB(ls_dict_X_train=new_KmerConstruct.all_freq_dict,n_features=len(new_KmerConstruct.features),y_train=genus_labels,alpha=0.1,include_prior=True) 
		model.train()
		
		model_filename='../models/NBmodel_'+ str(k) + 'mer.pickle'
		print('saving the file...')
		with open(model_filename,'wb') as handle:
			pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print('kmer freq for k=',k,' is done')


if __name__ == '__main__':
	ks=[4,6,7,8,9,10,11,12,14,16,18,32,64,100]
	n_threads=len(ks)
	pool = Pool(n_threads)
	pool.map(train,ks)

