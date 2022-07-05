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
from sklearn.ensemble import RandomForestRegressor
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
import requests
import json
import csv



st.set_page_config(page_title='Driving behaivor',page_icon=':truck:')
st.set_option('deprecation.showPyplotGlobalUse', False)
RF_model=pickle.load(open('RF_model.pkl','rb'))
DT_model=pickle.load(open('DT_model.pkl','rb'))
GB_model=pickle.load(open('GB_model.pkl','rb'))



with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2


def streamlit_menu(example=1):
    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Behaviour","Many traject","Graphs","Analyses"],  # required
            icons=["bi-truck","bi-file-text", "bi-graph-up","bi-pie-chart-fill"],  # optional
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

if selected == "Many traject":
    with open('style.css') as f:
     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
     st.header('''
     Classification of driving journeys for several days:
     ''')   
    st.sidebar.header("Select classifier")
    classifier_name = st.sidebar.selectbox(
         '',
         ('DT', 'GB','RF')
    )
    st.subheader('')
    def main():
        data_file = st.file_uploader("Upload driver data",type=['csv'])
        if st.button("Process"):
                if data_file is not None:
                    file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
                    df = pd.read_csv(data_file)
                else:
                    df = pd.DataFrame(columns=['DATE_DE_DEPART','PSN','CHAUFFEUR','TONNAGE','Anne_v'])    
        else :
            df = pd.DataFrame(columns=['DATE_DE_DEPART','PSN','CHAUFFEUR','TONNAGE','Anne_v'])
        return df

    df = main()
    if  not df.empty:
        df=df.copy().drop('N_Ordre', axis=1)
        df= df.drop_duplicates()
        df1= df[['DATE_DE_DEPART','PSN','CHAUFFEUR','TONNAGE','Anne_v']].copy()
        # st.write(df1)
        df.drop(['CHAUFFEUR','DATE_DE_DEPART','Anne_v','PSN','moy_alt'], axis=1, inplace=True)
        # st.write(df)
        df['TONNAGE']=[float(str(i).replace(",",""))for i in df["TONNAGE"]]
        def new_features(dff):
            dff['dist_to_fuel'] = dff['fuel_estim'] / dff['dist_estim'] 
            dff['moy_acc'] = dff['moy_acc']
            dff['moy_sog'] = dff['moy_sog'] / 50.0
            dff['TONNAGE'] = dff['TONNAGE'] / 50.0     
            return dff
        df = new_features(df)
        df2=df.copy().drop('fuel_estim', axis=1)
        df2=df2.copy().drop('dist_estim', axis=1)
        liste=['agressif driving(reduce your acceleration and increase your speed)',
        'agressif driving(reduce your acceleration)',
        'bad driving',
        'bad driving(increase your speed)',
        'bad driving(reduce your acceleration)',
        'best driving',
        'good driving',
        'good driving(increase your speed)',
        'good driving(reduce your acceleration)',
        'normal driving',
        'normal driving(increase your speed)',
        'normal driving(reduce your acceleration)']
        le=preprocessing.LabelEncoder()
        le.fit(liste)
        list(le.classes_)
        z=le.transform(liste)
        # st.write(z)
        def get_classifier(clf_name):
            clf = None  
            if clf_name == 'GB':
                p=GB_model.predict(df2)
                h=[]
                for i in p:
                    h = np.append(h,le.inverse_transform([int(i)]))
                df1['comportement de conduite']=h                
                st.write(df1)        
            elif clf_name == 'RF':
                st.subheader("The behaviour of the driver by Random Forests:") 
                p=RF_model.predict(df2)
                h=[]
                for i in p:
                    h = np.append(h,le.inverse_transform([int(i)]))
                df1['comportement de conduite']=h                
                st.write(df1)
            else:
                st.header("The behaviour of the driver by Decision Tree:") 
                p=DT_model.predict(df2)
                h=[]
                for i in p:
                    h = np.append(h,le.inverse_transform([int(i)]))

                df1['comportement de conduite']=h
                    
                st.write(df1)
            return clf
        clf = get_classifier(classifier_name)   
    ########
    css_example = '''                                                                                                                                                      
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

    <i class="fa-solid fa-truck" style="width:500px"></i>   
                                                                                                                                                                                                                                                                                   
    '''
    st.write(css_example, unsafe_allow_html=True) 




if selected == "Behaviour":
    st.title(f" {selected} of driver:")
    with open('style.css') as f:
     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.sidebar.header("Select classifier")
    def user_input():
        TONNAGE = st.number_input('',min_value=0,max_value=70)
        st.write('Tonnage ', TONNAGE )
        moy_acc = st.number_input('',key=3)
        st.write('Moy accélération  ', moy_acc)
        moy_sog = st.number_input('',key=5)
        st.write('Moy vitesse ', moy_sog)
        dist_estim = st.number_input('',key=1)
        st.write('Distance ', dist_estim)
        fuel_estim = st.number_input('',key=2)
        st.write('Fuel', fuel_estim)
        data={
        'TONNAGE':TONNAGE,
        'moy_acc':moy_acc,
        'moy_sog':moy_sog,
        'dist_estim':dist_estim,
        'fuel_estim':fuel_estim,
        }
        fuel_parametres=pd.DataFrame(data,index=[0])
        return fuel_parametres
    df1=user_input()

    classifier_name = st.sidebar.selectbox(
         '',
         ('DT', 'GB','RF')
    )
    st.subheader('')
    st.write(df1)
    if (df1 == 0).sum(axis=1).any():
        st.text('You must fill in all fields')
    else:
        liste=['agressif driving(reduce your acceleration and increase your speed)',
        'agressif driving(reduce your acceleration)',
        'bad driving',
        'bad driving(increase your speed)',
        'bad driving(reduce your acceleration)',
        'best driving',
        'good driving',
        'good driving(increase your speed)',
        'good driving(reduce your acceleration)',
        'normal driving',
        'normal driving(increase your speed)',
        'normal driving(reduce your acceleration)']
        le=preprocessing.LabelEncoder()
        le.fit(liste)
        list(le.classes_)
        z=le.transform(liste)
        def new_features_pred1(dff):
            dff['dist_to_fuel'] = dff['fuel_estim'] / dff['dist_estim']
            dff['moy_acc'] = dff['moy_acc']
            dff['moy_sog'] = dff['moy_sog'] / 50.0
            dff['TONNAGE'] = dff['TONNAGE'] / 50.0 
            dff=dff.copy().drop('fuel_estim', axis=1)
            dff=dff.copy().drop('dist_estim', axis=1)    
            return dff
        df1=new_features_pred1(df1)
        if df1['dist_to_fuel'].isnull().values.any():
            df1['dist_to_fuel']=0
        def get_classifier(clf_name):
            clf = None  
            if clf_name == 'GB':
                st.subheader("Le comportement du chaffeur par Gradient Boosting:") 
                p=GB_model.predict(df1)
                h=[]
                for i in p:
                    h = np.append(h,le.inverse_transform([int(i)]))
                df1['comportement de conduite']=h    
                st.text(h[0])
            elif clf_name == 'RF':
                st.subheader("Le comportement du chaffeur par Random Forests:") 
                p=RF_model.predict(df1)
                h=[]
                for i in p:
                    h = np.append(h,le.inverse_transform([int(i)]))
                df1['comportement de conduite']=h        
                st.text(h[0])
            else:
                st.header("Le comportement du chaffeur par arbre de décision:") 
                p=DT_model.predict(df1)
                h=[]
                for i in p:
                    h = np.append(h,le.inverse_transform([int(i)]))
                df1['comportement de conduite']=h        
                st.text(h[0])
            return clf
        clf = get_classifier(classifier_name)

        

    



    ########

    css_example = '''                                                                                                                                                      
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

    <i class="fa-solid fa-truck" style="width:500px"></i>   
                                                                                                                                                                                                                                                                                   
    '''

if selected == "Analyses":
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

    plt.pie(Tasks, labels = mylabels, explode = myexplode, shadow = True,autopct='%1.1f%%', textprops={'fontsize':16})
    plt.title('Relativistic circle of traject classes :',fontsize=20)
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
    plt.title('Graphic columns of traject classes :',fontsize=16)

    plt.bar(['NORMAL_N','GOOD_N','BAD_N','BEST_N','AGRESSIVE_N'],[c1,d1,e1,f1,g1],label='NEW',color="#699BCA",width=.6)
    plt.bar(['NORMAL_O','GOOD_O','BAD_O','BEST_O','AGRESSIVE_O'],[c2,d2,e2,f2,g2],label='OLD',color='#C4CDD8',width=.6)
    plt.legend()
    plt.ylabel('PERCENTAGE %',fontsize=16)
    
    fig2 = plt.show() 
    plt.xticks(rotation=45,fontsize=14)
    plt.yticks(rotation=45,fontsize=14)

    st.pyplot(fig2)

# pickle.dump(DT,open('DT_model.pkl','wb'))
# pickle.dump(RF,open('RF_model.pkl','wb'))
# pickle.dump(GB,open('GB_model.pkl','wb'))













