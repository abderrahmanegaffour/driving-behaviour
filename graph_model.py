import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn import tree
from IPython.display import display
import pickle

a ='A4BA0D85'
b ='2021-01-08T00:00:00'


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
    


# rslt_1 = df.loc[df['PSN'] == user_input_PSN() ]
# rslt_2 = rslt_1.loc[rslt_1['DATE_DE_DEPART'] == b]
rslt_2= df.loc[(df["PSN"] == a) & (df["DATE_DE_DEPART"] == b)]
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



pickle.dump(fig,open('Graph_model.pkl','wb'))
