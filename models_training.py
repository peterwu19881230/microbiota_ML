"""load X,y"""#Logistic Regression: failedfrom sklearn.linear_model import LogisticRegressionfrom utility import readFasta, get_label_list, get_accu_f1X=new_KmerConstruct.X#clf = LogisticRegression(random_state=0).fit(X[0:2000], y[0:2000]) #STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.#Decision tree: train fast yet inaccurate: Accuracy=  0.3191489361702128, F1=  0.4838709677419355from sklearn.tree import DecisionTreeClassifierdecisiontree = DecisionTreeClassifier(random_state=0)decisiontree.fit(X,y)#y_pred=decisiontree.predict(X)#get_accu_f1(y_pred,y)#neuro network