import numpy as np
from tqdm import tqdm

class tup_input_multinomialNB():
    def __init__(self,ls_tutple_X_train,n_features,y_train,alpha=0.1,include_prior=True):
        self.ls_tutple_X_train=ls_tutple_X_train
        self.n_features=n_features
        self.y_train=y_train
        self.alpha=alpha
        self.uniq_y=None
        self.include_prior=include_prior
        
    def train(self):
        self.all_prior_yi=[]
        self.all_p_feature_given_yi=[]
        self.uniq_y=sorted(list(set(self.y_train)))
        
        print('==Training...==')
        for yi in tqdm(self.uniq_y):
            
            ind=[i for i, _y in enumerate(self.y_train) if _y == yi]
            
            #calculate prior probabilities for each class
            if self.include_prior:
                self.all_prior_yi.append(len(ind)/len(self.y_train))
               
            #calculate p(a kmer|a class) for each class
            ls_tutple_X_yi=[self.ls_tutple_X_train[i] for i in ind]
            N_yi=sum([tup[1] for tup_ls in ls_tutple_X_yi for tup in tup_ls])
            p_feature_given_yi=[]
            for i in range(self.n_features): 
                sum_feature_vec=0
                for tup_ls in ls_tutple_X_yi:
                    for tup in tup_ls:
                        if tup[0]==i: sum_feature_vec+=tup[1]                           
                p_feature_given_yi.append((sum_feature_vec+self.alpha)/(N_yi+self.alpha*self.n_features))
            
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
                    if freq==0: freq+=self.alpha #not sure if this is the best smoothing
                    log_prob_yi+=freq*np.log(p_feature_given_yi) 
                log_prob_all_y.append(log_prob_yi)
            
            prediction_ind=log_prob_all_y.index(max(log_prob_all_y))
            prediction=self.uniq_y[prediction_ind]
            y_pred.append(prediction)
        
        return y_pred
