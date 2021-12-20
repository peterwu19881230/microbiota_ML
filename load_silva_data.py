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