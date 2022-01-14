from dict_input_multinomialNB import dict_input_multinomialNB
import pickle
from os.path import exists 
import numpy as np

if __name__ == '__main__':
    
    with open('load_silva_data.py') as infile:
        exec(infile.read()) #ss, genus_labels are given here
    
    for k in [4,6,7,8,9,10,11,12,14,16,18,32,64,100]:
        filename='../kmer_feature_objects/KmerConstruct_'+ str(k) + 'mer.pickle'
        if exists(filename):        
            new_KmerConstruct = pickle.load(open(filename,"rb" ))
            
            model=dict_input_multinomialNB(ls_dict_X_train=new_KmerConstruct.all_freq_dict,n_features=len(new_KmerConstruct.features),y_train=genus_labels,alpha=0.1,include_prior=True) 
            model.train()
            
            model_filename='../models/NBmodel_'+ str(k) + 'mer.pickle'
            print('saving the file...')
            with open(model_filename,'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('kmer freq for k=',k,' is done')