# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:48:30 2019

@author: Administrator
"""
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix ,classification_report,mean_squared_error,accuracy_score,roc_curve,auc
from sklearn import datasets 
iris = datasets.load_iris()
label=iris['target']
data=iris['data']
x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.33, random_state=0)


#le modele de classification en utilisant svm multiclasse (un contre un)
model=svm.SVC(kernel='linear',C=1, decision_function_shape='ovo')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#ploter la matrice de conf par sns.heatmap(pd.dataframe(cm))
cm=confusion_matrix(y_test,y_pred)
rep=classification_report(y_test,y_pred)
err=mean_squared_error(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)

""" Plot roc_curve n' as aucune signification pour les multiclasses  """
fpr,tpr,thresholds=roc_curve(y_test,y_pred,pos_label=1)
roc_auc=auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.plot (fpr,tpr,color='blue')
plt.title('Receiver operating characteristic' )
plt.plot(fpr,tpr,'b',label='AUC=%0.2f' %roc_auc)
plt.legend(loc='lower right ')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('True positive rate')
plt.ylabel('True negative rate')
plt.show()
# rq:auc et roc_curve on ne peut pas les ploter pour les cas multiclasses 

""" complexité de calcul c le temps ou le model converge """
import time 
start_time=time.time()
model.fit(x_train,y_train)
train_time=time.time()-start_time 

#le modele de classification en utilisant svm multiclasse (un contre all)
model_all=svm.SVC(kernel='linear',C=1, decision_function_shape='ovr')
model_all.fit(x_train,y_train)
y_pred_all=model_all.predict(x_test)
#ploter la matrice de conf par sns.heatmap(pd.dataframe(cm))
cm_all=confusion_matrix(y_test,y_pred)
rep_all=classification_report(y_test,y_pred)
err_all=mean_squared_error(y_test,y_pred)
acc_all=accuracy_score(y_test,y_pred)

""" complexité de calcul c le temps ou le model converge """
import time 
start_time=time.time()
model_all.fit(x_train,y_train)
train_time_all=time.time()-start_time 

"""-----kernel=rbf(c,gamma)--------------------------------------------------------"""
#le modele de classification en utilisant svm multiclasse (un contre un)
model=svm.SVC(kernel='rbf',C=1,gamma=0.1, decision_function_shape='ovo')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#ploter la matrice de conf par sns.heatmap(pd.dataframe(cm))
cm=confusion_matrix(y_test,y_pred)
rep=classification_report(y_test,y_pred)
err=mean_squared_error(y_test,y_pred)
acc_rbf=accuracy_score(y_test,y_pred)
"""-----kernel=sigmoid(coef0,gamma )---------------------------------------------------------""""
#le modele de classification en utilisant svm multiclasse (un contre un)
model=svm.SVC(kernel='sigmoid',gamma=0.02,coef0=0.1, decision_function_shape='ovo')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#ploter la matrice de conf par sns.heatmap(pd.dataframe(cm))
cm=confusion_matrix(y_test,y_pred)
rep=classification_report(y_test,y_pred)
err=mean_squared_error(y_test,y_pred)
acc_sig=accuracy_score(y_test,y_pred)
# rq: les fonctions OneVsOneClassifier et OneVsAllClassifier utilisent que le kernellinear linearclassifier 

# conclusion : les 3 kernels convergent rapidement vers  une acc 0.98 car les classes sont bien separées  
