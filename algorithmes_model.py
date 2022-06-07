from urllib.parse import uses_relative
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from IPython.display import display
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle



train_data = pd.read_csv('data/final_hl_dataset_all_eng.csv')
x = train_data.drop(columns=['N_Ordre','PSN','comportement de conduit','CHAUFFEUR'])
y = train_data['comportement de conduit']


# LR
sc_X = StandardScaler()
X_LR = sc_X.fit_transform(x)
LR = LogisticRegression(max_iter = 500000)
LR.fit(X_LR, y)


# SVM
SVM = make_pipeline(StandardScaler(), SVC(gamma='auto'))
SVM.fit(x,y)


# KNN
sc_X = StandardScaler()
X_KNN = sc_X.fit_transform(x)
KNN=KNeighborsClassifier(n_neighbors=7,metric='minkowski',p=2)
KNN.fit(X_KNN,y)

#Decision tree model
DT = DecisionTreeClassifier()
#train the model
DT.fit(x,y)

#Extra tree model
EXDT=ExtraTreesClassifier(criterion="entropy",random_state=0)
#train the model
EXDT.fit(x,y)

#RF model
RF = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=1)
#train the model
RF.fit(x,y)

# GB
GB = GaussianNB()
GB.fit(x, y)
#ask to make predictions with GB


pickle.dump(LR,open('LR_model.pkl','wb'))
pickle.dump(DT,open('DT_model.pkl','wb'))
pickle.dump(EXDT,open('EXDT_model.pkl','wb'))
pickle.dump(RF,open('RF_model.pkl','wb'))
pickle.dump(SVM,open('SVM_model.pkl','wb'))
pickle.dump(KNN,open('KNN_model.pkl','wb'))
pickle.dump(GB,open('GB_model.pkl','wb'))













