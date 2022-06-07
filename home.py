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
# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import graphviz
from IPython.display import display
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from streamlit_option_menu import option_menu
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

LR_model=pickle.load(open(r'C:\Users\abdou\Desktop\web_app\models\LR_model.pkl','rb'))
DT_model=pickle.load(open(r'C:\Users\abdou\Desktop\web_app\models\DT_model.pkl','rb'))
EXDT_model=pickle.load(open(r'C:\Users\abdou\Desktop\web_app\models\EXDT_model.pkl','rb'))
RF_model=pickle.load(open(r'C:\Users\abdou\Desktop\web_app\models\RF_model.pkl','rb'))
SVM_model=pickle.load(open(r'C:\Users\abdou\Desktop\web_app\models\SVM_model.pkl','rb'))
KNN_model=pickle.load(open(r'C:\Users\abdou\Desktop\web_app\models\KNN_model.pkl','rb'))
GB_model=pickle.load(open(r'C:\Users\abdou\Desktop\web_app\models\GB_model.pkl','rb'))


#fig=pickle.load(open('Graph_model.pkl','rb'))

st.set_page_config(page_title='Driving behaivor',page_icon=':truck:')
st.set_option('deprecation.showPyplotGlobalUse', False)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2


def streamlit_menu(example=1):
    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Driving behaviour", "Graphs"],  # required
            icons=["bi-truck", "bi-graph-up"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected
    


selected = streamlit_menu(example=EXAMPLE_NO)
if selected == "Graphs":
    st.title(f"{selected} of Driver:")

    def user_input_PSN():
        #PSN = st.text_input('',key=6)
        #st.write('PSN ', PSN)
        PSN = st.selectbox(
            'Select PSN',
            ('A4BA0D33', '85BA0BED', 'A4BA0D85','A4BA0CF9','A4BA0DE7','85BA0BFC','A4BA0CAC','00BA06C9',
            '85BA0BF9','A4BA0CD6','04BB0703','00BA07E6','20BA08C6','20BA08FA','85BA0BF4','85BA0BD1',
            '85BA0BD7','A4BA0E48','04BB0607','A4BA0D07','04BC0D82','04BB06E5','85BA0BDF',
            '85BA0BCF','04BB0454','85BA0BD9'))

        st.write('You selected:', PSN)

        dataPSN={'PSN':PSN,
        }

        PSN=dataPSN.get('PSN')
        return PSN

    def user_input_Date():
        Date = st.date_input('',key=7)
        st.write('Date ', Date),
        #time = st.time_input('Insert time of the trip')
        #st.write('time ', time),

        dataDate={'Date':Date.strftime('%Y-%m-%d')+'T00:00:00',
        }
        
        
        Date=dataDate.get('Date')
        return Date
        
    
    b =user_input_Date()
    c= user_input_PSN()
    print('value of c is',c)
    print(b)


    #st.pyplot(fig)


if selected == "Driving behaviour":
    st.title(f" {selected}:")
    with open('style.css') as f:
     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
     st.write('''
     # Classification des conducteurs:
     ''')

    st.sidebar.header("Select classifier")



    def user_input():
        Tonnage = st.number_input('',min_value=0,max_value=70)
        st.write('Tonnage ', Tonnage)
        Distance = st.number_input('',key=1)
        st.write('Distance ', Distance)
        Fuel = st.number_input('',key=2)
        st.write('Fuel', Fuel)
        Moy_acc = st.number_input('',key=3)
        st.write('Moy accélération  ', Moy_acc)
        Moy_alt = st.number_input('',key=4)
        st.write('Moy altitude ', Moy_alt)
        Moy_sog = st.number_input('',key=5)
        st.write('Moy vitesse ', Moy_sog)
        

        #Tonnage=st.sidebar.slider('Tonnage',min_value=0, max_value=70)
        #Distance=st.sidebar.slider('Distance',0.0,1000.0,200.0)
        #Fuel=st.sidebar.slider('Fuel',0.0,1000.0,200.0)
        #Moy_acc=st.sidebar.slider('Moy_acc',0.0,1.00,0.0)
        #Moy_alt=st.sidebar.slider('Moy_alt',100.0,1000.0,300.0)
        #Moy_sog=st.sidebar.slider('moy_sog',0.0,120.0,50.0)


        data={'Tonnage':Tonnage,
        'Distance':Distance,
        'Fuel':Fuel,
        'Moy_acc':Moy_acc,
        'Moy_alt':Moy_alt,
        'Moy_sog':Moy_sog,
        }
        fuel_parametres=pd.DataFrame(data,index=[0])
        return fuel_parametres


        #Tonnage = st.number_input('Insert Tonnage')
        #st.write('The current number is ', Tonnage)
        #Distance = st.number_input('Insert Distance')
        #st.write('The current number is ', Distance)

    df=user_input()

    predictionsRF = RF_model.predict(df)
    predictionsKNN = KNN_model.predict(df)
    predictionsGB = GB_model.predict(df)
    predictionsSVM = SVM_model.predict(df)
    predictionsLR = LR_model.predict(df)
    predictionsDT = DT_model.predict(df)
    predictionsEXDT = EXDT_model.predict(df)

    classifier_name = st.sidebar.selectbox(
         '',
         ('DT','LR','KNN', 'SVM', 'GB','EXDT','RF')
    )


    st.subheader('')
    st.write(df)

    def get_classifier(clf_name):
        clf = None
        if clf_name == 'SVM':
         
            st.subheader("Le comportement du chaffeur par Machine à vecteurs de support:") 
            st.text(predictionsSVM[0]) 

        elif clf_name == 'RF':
      
            st.subheader("Le comportement du chaffeur par Random Forests:") 
            st.text(predictionsRF[0]) 

        elif clf_name == 'KNN':
       
            st.subheader("Le comportement du chaffeur par K-Plus Proches-Voisins:") 
            st.text(predictionsKNN[0]) 

        elif clf_name == 'GB':
            st.subheader("Le comportement du chaffeur gradient boosting:") 
            st.text(predictionsGB[0])

        elif clf_name == 'EXDT':  
            st.subheader("Le comportement du chaffeur Extra DT:") 
            st.text(predictionsEXDT[0])    
        elif clf_name == 'LR':  
            st.subheader("Le comportement du chaffeur Logistic regression:") 
            st.text(predictionsLR[0])        
        else:
            st.header("Le comportement du chaffeur par arbre de décision:") 
            st.text(predictionsDT[0])

        return clf

    clf = get_classifier(classifier_name)

    





    ########

css_example = '''                                                                                                                                                      
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

    <i class="fa-solid fa-truck" style="width:500px"></i>   
                                                                                                                                                                                                                                                                                   
    '''
    

st.write(css_example, unsafe_allow_html=True) 
