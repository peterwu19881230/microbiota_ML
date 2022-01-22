#construct discrit mixed kmer obj based on discrit kmers from all kmers mixed

import pickle

new_KmerFeatureEng=pickle.load(open('../kmer_feature_objects/new_KmerFeatureEng_all_mer.pickle','rb'))
new_KmerFeatureEng.construct_discrit_freq_dict(reamin=.5)


with open('../kmer_feature_objects/new_KmerFeatureEng_all_mer_discrit_0.5.pickle','wb') as handle:
    pickle.dump(new_KmerFeatureEng, handle, protocol=pickle.HIGHEST_PROTOCOL)                
    print('done')   