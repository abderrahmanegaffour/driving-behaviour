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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle



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
        
        #foction de calcule de distances
    def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
        #if to_radians:
        #   lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

        a = np.sin((lat2-lat1)/2.0)**2 + \
            np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

        return earth_radius * 2 * np.arcsin(np.sqrt(a))

   # lire les données a partir de fichier dataset
    df = pd.read_csv('DATASET_ALL.csv' )
    df.head()
    #df = df.drop(['Unnamed: 0.1'], axis=1)
    #effacer les ligne dupliquées
    df1 = df[['PSN','DATE_DE_DEPART','CHAUFFEUR']].drop_duplicates()
    
    a = 'A4BA0D85'
    d='2021-01-07T00:00:00'
    b =user_input_Date()
    c= user_input_PSN()
    print('value of c is',c)
    print(b)


    # rslt_1 = df.loc[df['PSN'] == user_input_PSN() ]
    # rslt_2 = rslt_1.loc[rslt_1['DATE_DE_DEPART'] == b]
    rslt_2= df.loc[(df["PSN"] == c) & (df["DATE_DE_DEPART"] == b)]
    #drop rows inutiles (speed=0)
    rslt_2.drop(rslt_2[rslt_2['SOG'] == 0].index, inplace = True)

    #changemen de type de temp de string a datetime
    rslt_2['TIMESTAMP'] =pd.to_datetime(rslt_2['TIMESTAMP'])

    #les operations nécéssaires
    rslt_2['sog_dif'] = rslt_2['SOG'].diff().fillna(0)
    # rslt_2['temps_dif'] = pd.Series(rslt_2['TIMESTAMP'].view('int64').diff().fillna(0).div(1e9))

    rslt_2['temps_dif'] = pd.Series(rslt_2['TIMESTAMP'].astype('int64').diff().fillna(0).div(1e9))
    rslt_2['acceleration']=rslt_2['sog_dif']*0.28/rslt_2['temps_dif']
    rslt_2['LAT_rad'], rslt_2['LON_rad'] = np.radians(rslt_2['LAT']), np.radians(rslt_2['LON'])
    rslt_2['fuel_estim']= -0.00042379* pow(rslt_2['sog_dif'],2)+0.05886221*rslt_2['sog_dif']+0.88966832
    rslt_2['dist'] = \
        haversine(rslt_2.LAT_rad.shift(), rslt_2.LON_rad.shift(),
                    rslt_2.loc[1:, 'LAT_rad'], rslt_2.loc[1:, 'LON_rad'])


    conso=0
    df4 = pd.DataFrame(columns=['fuel_estim','consomation','TIMESTAMP'])
    for i,row in rslt_2.iterrows():
        conso=conso+row['fuel_estim']
        new_row1 = {'TIMESTAMP':row['TIMESTAMP'],'fuel_estim':row['fuel_estim'],'consomation':conso}
        df4 = df4.append(new_row1, ignore_index = True)


    rslt_2['consomation']=df4['consomation']







    
    # ici la figure
    plt.rcParams["figure.figsize"] = (20,10)

    x1 = np.array(rslt_2.SOG)
    y = np.array(rslt_2.TIMESTAMP)
    plt.plot(y, x1, color='red', linestyle='--',label="vitesse")

    x2 = np.array(rslt_2.acceleration*10)
    plt.plot(y, x2, label="acceleration")

    x3 = np.array(df4.consomation)
    plt.plot(y, x3 ,color='green', linestyle='--', label="consommation")
    plt.gca().legend(('vitesse','acceleration','consommation'))

    fig = plt.show()

    
    st.pyplot(fig)





if selected == "Driving behaviour":
    st.title(f" {selected}:")
    with open('style.css') as f:
     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
     st.write('''
     # Classification des conducteurs:
     ''')

    st.sidebar.header("Select classifier")



    def user_input():
        Tonnage = st.number_input('',min_value=0,max_value=70,key=4)
        st.write('Tonnage ', Tonnage)
        Moy_acc = st.number_input('',key=7)
        st.write('Moy accélération/décélération  ', Moy_acc)
        Moy_sog = st.number_input('',key=9)
        st.write('Moy vitesse ', Moy_sog)
        Moy_alt = st.number_input('',key=8)
        st.write('Moy altitude ', Moy_alt)
        Distance = st.number_input('',key=5)
        st.write('Distance ', Distance)
        Fuel = st.number_input('',key=6)
        st.write('Fuel', Fuel)
        

        #Tonnage=st.sidebar.slider('Tonnage',min_value=0, max_value=70)
        #Distance=st.sidebar.slider('Distance',0.0,1000.0,200.0)
        #Fuel=st.sidebar.slider('Fuel',0.0,1000.0,200.0)
        #Moy_acc=st.sidebar.slider('Moy_acc',0.0,1.00,0.0)
        #Moy_alt=st.sidebar.slider('Moy_alt',100.0,1000.0,300.0)
        #Moy_sog=st.sidebar.slider('moy_sog',0.0,120.0,50.0)


        
        data={
        'Tonnage':Tonnage,
        'Moy_acc':Moy_acc,
        'Moy_sog':Moy_sog,
        'Moy_alt':Moy_alt,
        'Distance':Distance,
        'Fuel':Fuel,
        }
        fuel_parametres=pd.DataFrame(data,index=[0])
        return fuel_parametres


        #Tonnage = st.number_input('Insert Tonnage')
        #st.write('The current number is ', Tonnage)
        #Distance = st.number_input('Insert Distance')
        #st.write('The current number is ', Distance)

    df=user_input()

    classifier_name = st.sidebar.selectbox(
         '',
         ('DT','LR','KNN', 'SVM', 'GB','EXDT','RF')
    )


    st.subheader('')
    st.write(df)

    train_data = pd.read_csv('final_dataset.csv', encoding="latin1")
    x = train_data.drop(columns=['N_Ordre','DATE_DE_DEPART','CHAUFFEUR','comportement de conduit'])
    y = train_data['comportement de conduit']

    # SVM
    SVM = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    SVM.fit(x,y)
    #ask to make predictions with GB
    predictionsSVM = SVM.predict(df)


    # KNN
    #KNN = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    #KNN.fit(x,y)
    #predictionsKNN=KNN.predict(df)

    train_data = pd.read_csv('final_dataset.csv', encoding="latin1")
    x = train_data.drop(columns=['N_Ordre','DATE_DE_DEPART','CHAUFFEUR','comportement de conduit'])
    y = train_data['comportement de conduit']

    #Decision tree model
    DT = DecisionTreeClassifier()
    #train the model
    DT.fit(x,y)
    predictionsDT = DT.predict(df)
    #display(graphviz.Source(tree.export_graphviz(DT)))


    train_data = pd.read_csv('final_dataset.csv', encoding="latin1")
    x = train_data.drop(columns=['N_Ordre','DATE_DE_DEPART','CHAUFFEUR','comportement de conduit'])
    y = train_data['comportement de conduit']
    #Extra tree model
    EXDT=ExtraTreesClassifier(criterion="entropy",random_state=0)
    #train the model
    EXDT.fit(x,y)
    predictionsEXDT = EXDT.predict(df)


    train_data = pd.read_csv('final_dataset.csv', encoding="latin1")
    x = train_data.drop(columns=['N_Ordre','DATE_DE_DEPART','CHAUFFEUR','comportement de conduit'])
    y = train_data['comportement de conduit']

    #RF model
    RF = RandomForestClassifier(n_estimators=200, max_depth=200, random_state=1)
    #train the model
    RF.fit(x,y)
    predictionsRF = RF.predict(df)

    train_data = pd.read_csv('final_dataset.csv', encoding="latin1")
    x = train_data.drop(columns=['N_Ordre','DATE_DE_DEPART','CHAUFFEUR','comportement de conduit'])
    y = train_data['comportement de conduit']


    # GB
    sc_X = StandardScaler()
    X_GB = sc_X.fit_transform(x)
    GB = GradientBoostingClassifier()
    GB.fit(X_GB, y)
    #ask to make predictions with GB
    predictionsGB = GB.predict(df)

    train_data = pd.read_csv('final_dataset.csv', encoding="latin1")
    x = train_data.drop(columns=['N_Ordre','DATE_DE_DEPART','CHAUFFEUR','comportement de conduit'])
    y = train_data['comportement de conduit']
    # AB
    sc_X = StandardScaler()
    X_AB = sc_X.fit_transform(x)
    AB = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100, random_state=0)
    AB.fit(X_AB, y)
    #ask to make predictions with GB
    predictionsAB = AB.predict(df)

    #def add_parameter_ui(clf_name):
        #params = dict()
        #if clf_name == 'SVM':
            #C = st.sidebar.slider('C', 0.01, 20.0)
            #params['C'] = C
        #elif clf_name == 'KNN':
            #K = st.sidebar.slider('K', 1, 15)
            #params['K'] = K
            #clf_name == 'RF'
            #n_estimators = st.sidebar.slider('n_estimators', 10.0, 100.0)
            #params['n_estimators'] = n_estimators
        #return params

    #params = add_parameter_ui(classifier_name)


    def get_classifier(clf_name):
        clf = None
        if clf_name == 'SVM':
            #SVM model
            #SVM = svm.SVC(C=params['C'])
            #SVM.fit(x,y)
            #ask to make predictions with SVM
            #predictionsSVM = SVM.predict(df)
            #clf = svm.SVC(C=params['C'])
            st.subheader("Le comportement du chaffeur par Machine à vecteurs de support:") 
            st.text(predictionsSVM[0])


        #elif clf_name == 'KNN':
            #KNN model
            #KNN = KNeighborsClassifier(n_neighbors=params['K'])
            #train the model
            #KNN.fit(x,y)
            #ask to make predictions with KNN
            #predictionsKNN = KNN.predict(df)
            #clf = KNeighborsClassifier(n_neighbors=params['K'])
            #st.subheader("Le comportement du chaffeur par K-Plus Proches-Voisins:") 
            #st.text(predictionsKNN[0]) 

        elif clf_name == 'GB':
            st.subheader("Le comportement du chaffeur gradient boosting:") 
            st.text(predictionsGB[0])

        elif clf_name == 'AB':  
            st.subheader("Le comportement du chaffeur Ada Boost:") 
            st.text(predictionsAB[0])    

        elif clf_name == 'EXDT':  
            st.subheader("Le comportement du chaffeur Extra DT:") 
            st.text(predictionsEXDT[0])   
        
        elif clf_name == 'RF':
            #RF model
         #   RF = RandomForestClassifier(n_estimators=params['n_estimators'])
            #train the model
          #  RF.fit(x,y)
            #ask to make predictions with KNN
           # predictionsRF = RF.predict(df)
           # clf = RandomForestClassifier(n_estimators=params['n_estimators'])
            st.subheader("Le comportement du chaffeur par Random Forests:") 
            st.text(predictionsRF[0]) 

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

pickle.dump(AB,open('AB_model.pkl','wb'))
pickle.dump(DT,open('DT_model.pkl','wb'))
pickle.dump(EXDT,open('EXDT_model.pkl','wb'))
pickle.dump(RF,open('RF_model.pkl','wb'))
pickle.dump(SVM,open('SVM_model.pkl','wb'))
#pickle.dump(KNN,open('KNN_model.pkl','wb'))
pickle.dump(GB,open('GB_model.pkl','wb'))













