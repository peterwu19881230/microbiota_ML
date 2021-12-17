#read fasta files and return 2 lists: labels, sequences
def readFasta(filename):
    labels=[]
    seqs = []
    with open(filename, 'r') as f:
        for line in f:
            line=line.rstrip()
            if line[0] == '>':
                labels.append(line[1:])
            else:
                seqs.append(line)
    return labels, seqs 




def get_label_list(labels,type_='g__'):
    
    ##get genus as labels'
    type_labels=[]
    for label in labels:
        
        label_list=label.split(';')
        
        for label_ in label_list:
            if label_.find(type_)!=-1:
                type_labels.append(label_.lstrip()) #.lstrip() removes the space at the beginning
                break
       
        if label_.find(type_)==-1: #if the last label_ is still not what I want
            #print("this is not found:",type_)
            #print("Appending None for",type_)
            type_labels.append('None')
            
    return type_labels