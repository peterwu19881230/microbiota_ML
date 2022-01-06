"""
This NB model allows list of dictionaries as input
"""

import numpy as np
from tqdm import tqdm

class dict_input_multinomialNB():
    def __init__(self,ls_dict_X_train,n_features,y_train,alpha=0.1,include_prior=True):
        self.ls_dict_X_train=ls_dict_X_train
        self.n_features=n_features
        self.y_train=y_train
        self.alpha=alpha
        self.uniq_y=None
        self.include_prior=include_prior
        self.kmer_not_present_logp=[]
        
    def train(self):
        self.all_log_prior_yi=[]
        self.all_logp_feature_given_yi=[]
        self.uniq_y=sorted(list(set(self.y_train)))
        self.N_yi_ls=[]
        
        print('==Training...==')
        for i_,yi in tqdm(enumerate(self.uniq_y)):
            
            ind=[i for i, _y in enumerate(self.y_train) if _y == yi]
            
            #calculate prior probabilities for each class
            if self.include_prior:
                prob=len(ind)/len(self.y_train)
                self.all_log_prior_yi.append(np.log(prob))
               
            #calculate p(a kmer|a class) for each class
            ls_dict_X_train_yi=[self.ls_dict_X_train[i] for i in ind]
            N_yi=0
            sum_feature_vec_dict={}
            for dict_ in ls_dict_X_train_yi:              
                for kmer in dict_:
                    
                    N_yi+=dict_[kmer]
                    
                
                    if kmer not in sum_feature_vec_dict:
                        sum_feature_vec_dict[kmer]=dict_[kmer]
                    else:
                        sum_feature_vec_dict[kmer]+=dict_[kmer]
                        
            self.N_yi_ls.append(N_yi)
            
            logp_feature_given_yi={}
            for kmer in sum_feature_vec_dict:
                prob=(sum_feature_vec_dict[kmer]+self.alpha)/(N_yi+self.alpha*self.n_features) 
                logp_feature_given_yi[kmer]=np.log(prob)           
            
            self.all_logp_feature_given_yi.append(logp_feature_given_yi)
            self.all_unique_kmers=sorted(list(set([key for dict_ in self.all_logp_feature_given_yi for key in dict_])))
            
            prob=(self.alpha)/(self.N_yi_ls[i_]+self.alpha*self.n_features)
            self.kmer_not_present_logp.append(np.log(prob))
            
        self.ls_dict_X_train='-removed to save space-'
            
        print('\n')
        print('-------------')
        print('training done')
        print('-------------')
        
        
    def test(self,ls_dict_X_test):
        
        y_pred=[]
        for test_xi in tqdm(ls_dict_X_test): 
            log_prob_all_y=[]
            for i,yi in enumerate(self.uniq_y): #for each class    
                if self.include_prior: 
                    log_prob_yi=self.all_log_prior_yi[i]
                else:
                    log_prob_yi=0
                   
                for kmer in test_xi:
                    freq=test_xi[kmer]
                    
                    if kmer not in self.all_logp_feature_given_yi[i]:
                        logp=self.kmer_not_present_logp[i]
                    else:
                        logp=self.all_logp_feature_given_yi[i][kmer]
                        
                    log_prob_yi+=freq*logp
                                
                log_prob_all_y.append(log_prob_yi)
            
            prediction_ind=log_prob_all_y.index(max(log_prob_all_y))
            prediction=self.uniq_y[prediction_ind]
            y_pred.append(prediction)
        return y_pred
