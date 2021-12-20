#this might be changed to a jupyter notebook later


import matplotlib.pyplot as plt

import numpy as np
execfile('load_silva_data.py')

#Frequency table of genus

genus_freq_table={}
for label in genus_labels:
    if label not in genus_freq_table:
        genus_freq_table[label]=1
    else:
        genus_freq_table[label]+=1


#convert dictionary to list and plot the requency table
genus_freq_table_list=[]
for key in genus_freq_table:
   genus_freq_table_list.append(genus_freq_table[key])
genus_freq_table_list.sort()


plt.hist(np.array(genus_freq_table_list), 50, density=True, facecolor='g', alpha=0.75)


