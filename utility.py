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


def get_f1(answer_list,prediction_list):
    #The following is the same as implemented in the 2018 q2-classifier paper
    #====================
    set1=set(answer_list)
    set2=set(prediction_list)
    
    TP=len(set1 & set2)
    FP=len(set2.difference(set1))
    FN=len(set1.difference(set2))
    #====================
    
    f1=TP/(TP+0.5*(FP+FN))
    
    #print("F1= ",f1)
    
    return f1



def get_accu_f1(answer_list,prediction_list):
    #The following is the same as implemented in the 2018 q2-classifier paper
    #====================
    set1=set(answer_list)
    set2=set(prediction_list)
    
    TP=len(set1 & set2)
    TN= 0
    FP=len(set2.difference(set1))
    FN=len(set1.difference(set2))
    #====================
    
    accu=(TP+TN)/(TP+TN+FP+FN)
    f1=TP/(TP+0.5*(FP+FN))
    
    print("Accuracy= ",accu) 
    print("F1= ",f1)
    
    return accu,f1