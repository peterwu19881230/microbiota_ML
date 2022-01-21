import pickle
from os.path import exists 
from tqdm import tqdm

#get kmer feature lists
all_kmer_feature_lists=[]

for k in [4,6,7,8,9,10,11,12,14,16,18,32,64,100]:
    
    
    filename='../kmer_feature_objects/KmerFeatureEng_'+ str(k) + 'mer.pickle'
    if exists(filename):  
        print('mixing k=',k,'...\n')
        new_KmerConstruct = pickle.load(open(filename,"rb" ))
        all_kmer_feature_lists.append(new_KmerConstruct.all_freq_dict)
        
mixed_feature_lists=[]
no_samples=len(new_KmerConstruct.all_freq_dict)
for i in tqdm(range(no_samples)): #tqdm doesn't show up here. Don't know why
    mixed_feature_dict={}
    for kmer_feature_lists in all_kmer_feature_lists:
        mixed_feature_dict={**mixed_feature_dict,**kmer_feature_lists[i]}
        
    mixed_feature_lists.append(mixed_feature_dict)
    


filename='../kmer_feature_objects/NBmodel_mixed_mer.pickle'
print('saving the file...')
with open(filename,'wb') as handle:
    pickle.dump(mixed_feature_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('done')
