"""
Alternative implementation of sklearn multinomialNB
"""

import numpy as np
from tqdm import tqdm

class multinomialNB():
    def __init__(self,X_train,y_train,alpha=0.1,include_prior=True):
        self.X_train=X_train
        self.y_train=y_train
        self.alpha=alpha
        self.uniq_y=None
        self.include_prior=include_prior
        
    def train(self):
        self.all_prior_yi=[]
        self.all_p_feature_given_yi=[]
        n_features=self.X_train.shape[1]
        self.uniq_y=sorted(list(set(self.y_train)))
        
        print('==Training...==')
        for yi in tqdm(self.uniq_y):
            
            ind=[i for i, _y in enumerate(self.y_train) if _y == yi]
            
            #calculate prior probabilities for each class
            if self.include_prior:
                self.all_prior_yi.append(len(ind)/len(self.y_train))
               
            #calculate p(a kmer|a class) for each class
            X_yi=self.X_train[ind]
            N_yi=np.sum(X_yi)
            p_feature_given_yi=[]
            for i in range(X_yi.shape[1]):
                feature_vec=X_yi[:,i].A
                p_feature_given_yi.append((np.sum(feature_vec)+self.alpha)/(N_yi+self.alpha*n_features))
            
            self.all_p_feature_given_yi.append(p_feature_given_yi)
        print('\n')
        print('-------------')
        print('training done')
        print('-------------')
        
    def test(self,X_test):
        y_pred=[]
        for test_xi in tqdm(X_test):
            log_prob_all_y=[]
            for i,yi in enumerate(self.uniq_y): #for each class    
                if self.include_prior: 
                    log_prob_yi=np.log(self.all_prior_yi[i])    
                else:
                    log_prob_yi=0
                for j,p_feature_given_yi in enumerate(self.all_p_feature_given_yi[i]): #for each feature
                    freq=test_xi[j]                   
                    if freq!=0: #If freq=0 just ignore adding the log likelyhood
                        log_prob_yi+=freq*np.log(p_feature_given_yi) 
                        
                log_prob_all_y.append(log_prob_yi)
            
            prediction_ind=log_prob_all_y.index(max(log_prob_all_y))
            prediction=self.uniq_y[prediction_ind]
            y_pred.append(prediction)
        return y_pred
