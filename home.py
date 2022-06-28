from urllib.parse import uses_relative
from dataclasses import dataclass
import requests
import json
import csv
from scipy import stats
from scipy.stats import norm
from scipy.stats import skew
from pandas import json_normalize
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





st.set_page_config(page_title='Driving behaivor',page_icon=':truck:')
st.set_option('deprecation.showPyplotGlobalUse', False)


DT_model=pickle.load(open('DT_model.pkl','rb'))
EXDT_model=pickle.load(open('EXDT_model.pkl','rb'))
RF_model=pickle.load(open('RF_model.pkl','rb'))
SVM_model=pickle.load(open('SVM_model.pkl','rb'))
GB_model=pickle.load(open('GB_model.pkl','rb'))
AB_model=pickle.load(open('AB_model.pkl','rb'))
#KNN_model=pickle.load(open('KNN_model.pkl','rb'))



#fig=pickle.load(open('Graph_model.pkl','rb'))


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2


def streamlit_menu(example=1):
    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Driving behaviour","Graphs","Driver Analyses"],  # required
            icons=["bi-truck", "bi-graph-up","bi-pie-chart-fill"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            key=0
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
            '85BA0BCF','04BB0454','85BA0BD9'),key=1)

        st.write('You selected:', PSN)

        dataPSN={'PSN':PSN,
        }

        PSN=dataPSN.get('PSN')
        return PSN

    def user_input_Date():
        Date = st.date_input('',key=2)
        st.write('Date ', Date),
        #time = st.time_input('Insert time of the trip')
        #st.write('time ', time),

        dataDate={'Date':Date.strftime('%Y-%m-%d')+'T00:00:00',
                    'fin':Date.strftime('%Y-%m-%d')+'T23:59:59',
        }
        
        
        Date=dataDate.get('Date')
        fin= dataDate.get('fin')
        return Date,fin
        
    
        #foction de calcule de distances
    def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
        #if to_radians:
        #   lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

        a = np.sin((lat2-lat1)/2.0)**2 + \
            np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

        return earth_radius * 2 * np.arcsin(np.sqrt(a))

   # lire les données a partir de fichier dataset
    b =user_input_Date()
    c= user_input_PSN()


    df6 = pd.DataFrame(columns=['TIMESTAMP','ALT','LAT','LON' , 'SOG','DATE_DE_DEPART'])

    url = "https://idegps.com/mw/psn/"+c+"/record/16?key=nLsBvp0YedcXUcaU73fs&from_date="+b[0]+"&to_date="+b[1]
    print("search by psn")
    print(url)
    response = requests.get(url)
    response_json = response.json()
    dictionary = json.dumps(response.json(), sort_keys = True, indent = 4)
    df33 = pd.read_json(dictionary)    
    if not df33.empty:        
        df3 = df33.iloc[:, [25,0,11,12,22,1]].copy()
        print(df3)
    else:
        print(url)
        df3 = pd.DataFrame()
    
    df3.drop(df3[df3['SOG'] == 0].index, inplace = True)

    df3['TIMESTAMP'] =pd.to_datetime(df3['TIMESTAMP'])

    df3['sog_dif'] = df3['SOG'].diff().fillna(0)
    df3['temps_dif'] = pd.Series(df3['TIMESTAMP'].astype('int64').diff().fillna(0).div(1e9))
    df3['acceleration']=df3['sog_dif']*0.28/df3['temps_dif']
    df3['LAT_rad'], df3['LON_rad'] = np.radians(df3['LAT']), np.radians(df3['LON'])
    df3['fuel_estim']= -0.00042379* pow(df3['sog_dif'],2)+0.05886221*df3['sog_dif']+0.88966832
    df3['dist'] = \
        haversine(df3.LAT_rad.shift(), df3.LON_rad.shift(),
                 df3.loc[1:, 'LAT_rad'], df3.loc[1:, 'LON_rad'])


    conso=0
    df7 = pd.DataFrame(columns=['fuel_estim','consomation'])
    for i,row in df3.iterrows():
        conso=conso+row['fuel_estim']
        new_row1 = {'TIMESTAMP':row['TIMESTAMP'],'fuel_estim':row['fuel_estim'],'consomation':conso}
        df7 = df7.append(new_row1, ignore_index = True)

    df3['consomation']=df7['consomation']
    moy_sog= df3['SOG'].mean()
    moy_acc= df3['acceleration'].mean()
    dist_estim=df3['dist'].sum()
    fuel_estim=df3['fuel_estim'].sum()
    
    plt.rcParams["figure.figsize"] = (20,10)

    x1 = np.array(df3.SOG)

    y = np.array(df3.TIMESTAMP)
    plt.plot(y, x1, color='red', linestyle='--')

    x2 = np.array(df3.acceleration*10)
    plt.plot(y, x2)
    y = np.array(df3.TIMESTAMP)

    x3 = np.array(df7.consomation)


    plt.plot(y, x3 ,color='green', linestyle='--')
    plt.gca().legend(('Speed','Acceleration','Consumption'))

    fig = plt.show()

    
    st.pyplot(fig)


if selected == "Driver Analyses":
    st.title(f" {selected}:")
    with open('style.css') as f:
     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
     df4 = pd.read_csv('driver_analyses.csv', encoding='latin1' )
     


    def user_input_CHAUFFEUR():
        #CHAUFFEUR = st.text_input('',placeholder='Entrer le nom du chaffeur',key=3)
        CHAUFFEUR = st.selectbox(
            'SELECT DRIVER',
            ('TEFIANI MOHAMED','KRABIA ISHAK', 'EZZINE AHMED', 'SENOUCI SID AHMED','TEFIANI KOUIDER','BENAISSA AZZEDDINE',
            'TIDJINI KHELIFA','BENCHENI LAHCEN','KRIM AHMED','BELHADJ ALI','ABBOU ABD EL KADER','NOUALA MOKHTAR',
            'BOUAMAMA AMINE','HANSALI MOHAMED','SAHRAOUI MUSTAPHA','MESTARI MOHAMED',
            'TEFIANI YOUCEF','GAILI SAID','OUZAA ALI','HAMRI ABD EL KADER','DJELLAL MOHAMED','HAMIANI TAIEB',
            'NEHARI ABBES','KADDOURI BOUAZA','HAKEM LAKHDAR','BOUZIDI CHEIKH','BOUTAIBANE ABD EL KADER','ZIANE ABDELKADER',
            'SENNOUR MOHAMED ABDENACEUR','MAAZOUZ ABD EL HAK','SAHOUADJ RACHID','CHERCHAB BENDIDA',
            'LABOU HICHEM','TSAKI ABDERRAHMANE','BENIA MOHAMED OKACHA','HARCHI MUSTAPHA','RACHED BACHIR',
            'TEBIB ADEM','BEN ADDA LYES','ABBAD DJELLOUL','BENHAMOU MOHAMED','ROZAL ALI','AOUED MEHADJI',
            'LAREDJ BADER EDDINE',
            ),key = 12)
        

        st.write('Driver selected : ',CHAUFFEUR)

        dataCHAUFFEUR={'CHAUFFEUR':CHAUFFEUR,
        }

        CHAUFFEUR=dataCHAUFFEUR.get('CHAUFFEUR')
        return CHAUFFEUR
    e=user_input_CHAUFFEUR()
    a=df4.loc[df4['CHAUFFER']==e]
    Tasks = [a['AGRESSIVE CLASSE %'].item(),a['BAD CLASSE %'].item(),a['NORMAL CLASSE %'].item(),a['GOOD CLASSE %'].item(),a['BEST CLASSE %'].item()]
    mylabels = ["Agressive", "Bad","Normal", "Good", "Best "]
    myexplode = [0,0.3,0,0.1,0.1]

    plt.pie(Tasks, labels = mylabels, explode = myexplode, shadow = True,autopct='%1.1f%%', textprops={'fontsize': 14})
    plt.title('Relativistic circle of traject classes :',fontsize=16)
    print(5 * "\n")
    plt.axis('equal')
    fig = plt.show() 
    st.pyplot(fig)
    df5 = pd.read_csv('driver_classes_according_type.csv', encoding='latin1' )

    CHAUF=e
    b=df5.loc[df5['CHAUFFER']==CHAUF]
    a1=b.iloc[0]
    c1=a1['NORMAL CLASSE %']
    d1=a1['GOOD CLASSE %']
    e1=a1['BAD CLASSE %']
    f1=a1['BEST CLASSE %']
    g1=a1['AGRESSIVE CLASSE %']

    a2=b.iloc[1]
    c2=a2['NORMAL CLASSE %']
    d2=a2['GOOD CLASSE %']
    e2=a2['BAD CLASSE %']
    f2=a2['BEST CLASSE %']
    g2=a2['AGRESSIVE CLASSE %']
    print('nouvau: ',c1,d1,e1,f1,g1,' ancien: ',c2,d2,e2,f2,g2)
    plt.title('Graphic columns of traject classes :',fontsize=10)

    plt.bar(['NORMAL_N','GOOD_N','BAD_N','BEST_N','AGRESSIVE_N'],[c1,d1,e1,f1,g1],label='NEW CAR',color="#699BCA",width=.8)
    plt.bar(['NORMAL_O','GOOD_O','BAD_O','BEST_O','AGRESSIVE_O'],[c2,d2,e2,f2,g2],label='OLD CAR',color='#C4CDD8',width=.8)
    plt.legend()
    plt.ylabel('PERCENTAGE %')
    
    fig2 = plt.show() 
    plt.xticks(rotation=45,fontsize=6)
    plt.yticks(rotation=45,fontsize=6)

    st.pyplot(fig2)


if selected == "Driving behaviour":
    st.title(f" {selected}:")
    with open('style.css') as f:
     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
     st.write('''
     # Drivers classification: 
     ''',fontsize=12)

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

    predictionsRF = RF_model.predict(df)
    predictionsDT = DT_model.predict(df)
    predictionsGB = GB_model.predict(df)
    predictionsEXDT = EXDT_model.predict(df)
    predictionsSVM = SVM_model.predict(df)
    predictionsAB = AB_model.predict(df)
    #predictionsKNN = KNN_model.predict(df)

    
    
    

    classifier_name = st.sidebar.selectbox(
         '',
         ('DT','EXDT','RF','GB','AB','SVM'),key=10
    )


    st.subheader('')
    st.write(df)

    def get_classifier(clf_name):
        clf = None
        if clf_name == 'SVM':
         
            st.subheader("The behavior of the driver by Support Vector Machine:") 
            st.text(predictionsSVM[0]) 

        #elif clf_name == 'KNN':
       
            #st.subheader("Le comportement du chaffeur par K-Plus Proches-Voisins:") 
            #st.text(predictionsKNN[0]) 

        elif clf_name == 'GB':
            st.subheader("The behavior of the driver by Gradient Boosting:") 
            st.text(predictionsGB[0])

        elif clf_name == 'AB':  
            st.subheader("The behavior of the driver by Ada Boost:") 
            st.text(predictionsAB[0]) 

        elif clf_name == 'EXDT':  
            st.subheader("The behavior of the driver by Extra DT:") 
            st.text(predictionsEXDT[0])  
        elif clf_name == 'RF':
      
            st.subheader("The behavior of the driver by Random Forests:") 
            st.text(predictionsRF[0])   
       
        else:
            st.header("The behavior of the driver by Decision Tree:") 
            st.text(predictionsDT[0])

        return clf

    clf = get_classifier(classifier_name)

    





    ########

css_example = '''                                                                                                                                                      
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
                                                                                                                                                                                                                                                                                   
    '''
    

st.write(css_example, unsafe_allow_html=True) 
