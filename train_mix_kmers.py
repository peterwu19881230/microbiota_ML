from dict_input_multinomialNB import dict_input_multinomialNB
import pickle
import numpy as np
from os.path import exists 
from tqdm import tqdm



if __name__ == '__main__':
    
    with open('load_silva_data.py') as infile:
        exec(infile.read()) #ss, genus_labels are given here   

    filename='../kmer_feature_objects/KmerFeatureEng_all_mer.pickle'
    if exists(filename):        
        all_freq_dict = pickle.load(open(filename,"rb" ))
        
        print('calculating n_features')
        n_features=len(set([key for dict_ in tqdm(all_freq_dict) for key in dict_]))
        
        
        model=dict_input_multinomialNB(ls_dict_X_train=all_freq_dict,n_features=n_features,y_train=genus_labels,alpha=0.1,include_prior=True) 
        model.train()
        
        model_filename='../models/NBmodel_mix_kmer.pickle'
        print('saving the file...')
        with open(model_filename,'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('done')
    else:
    	print('training data not found!')