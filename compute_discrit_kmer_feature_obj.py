#construct discrit mixed kmer obj based on discrit kmers from all kmers mixed
from test_data import * #ss, genus_labels are given here
from KmerFeatureEng import KmerFeatureEng
import pickle

print('loading all_freq_dict')
all_freq_dict=pickle.load(open('../kmer_feature_objects/mixed_feature_lists.pickle','rb')) 


all_KmerFeatureEng=KmerFeatureEng(genus_labels,ss,k=None)
all_KmerFeatureEng.all_freq_dict=all_freq_dict


ss_kmers=[]
for dict_ in all_freq_dict:
    for key in dict_:
        for i in range(dict_[key]):
            ss_kmers.append(key)
all_KmerFeatureEng.ss_kmers=ss_kmers

all_KmerFeatureEng.features=all_KmerFeatureEng.get_all_uniq_kmers(all_KmerFeatureEng.ss_kmers)


remain=.5
print('constructing discrit freq dict')
all_KmerFeatureEng.construct_discrit_freq_dict(remain=remain)
discrit_freq_dict=all_KmerFeatureEng.discrit_freq_dict



with open('../kmer_feature_objects/new_KmerFeatureEng_all_mer_discrit_'+str(remain)+'.pickle','wb') as handle:
    pickle.dump(discrit_freq_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)                
    print('done')   