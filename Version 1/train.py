
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import math
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")
from datetime import datetime, date, time
from scipy import stats
from scipy.stats import pearsonr


# In[2]:


InsulinData = pd.read_csv("InsulinData.csv", header=None, sep=",")
InsulinData = InsulinData.drop(0)
InsulinData = InsulinData[[1,2,24]]
InsulinData.columns =['Date', 'Time', 'Meal']
InsulinData = InsulinData.reindex(index=InsulinData.index[::-1])
# print(InsulinData)


# In[3]:


Insulin_patient2 = pd.read_csv("Insulin_patient2.csv", header=None, sep=",")
Insulin_patient2 = Insulin_patient2.drop(0)
Insulin_patient2 = Insulin_patient2[[1,2,25]]
Insulin_patient2.columns =['Date', 'Time', 'Meal']
Insulin_patient2 = Insulin_patient2.reindex(index=Insulin_patient2.index[::-1])
# print(Insulin_patient2)


# In[4]:


CGMData = pd.read_csv("CGMData.csv", sep=",", header = None)
CGMData = CGMData.drop(0)
CGMData = CGMData[[1,2,30]]
CGMData.columns =['Date', 'Time', 'Glucose']
CGMData = CGMData.reindex(index=CGMData.index[::-1])
# print(CGMData)


# In[5]:


CGM_patient2 = pd.read_csv("CGM_patient2.csv", header=None, sep=",")
CGM_patient2 = CGM_patient2.drop(0)
CGM_patient2 = CGM_patient2[[1,2,31]]
CGM_patient2.columns =['Date', 'Time', 'Glucose']
CGM_patient2 = CGM_patient2.reindex(index=CGM_patient2.index[::-1])
# print(CGM_patient2)


# In[6]:


def isNaN(num):
    return num != num
for index, row in InsulinData.iterrows():
    i_nans = isNaN(row['Meal'])
    if( i_nans ):
        InsulinData = InsulinData.drop(index)
    if( row['Meal'] == '0' ):
        InsulinData = InsulinData.drop(index)
    if( row['Meal'] == 0 ):
        InsulinData = InsulinData.drop(index)      
# InsulinData.head()
# print(InsulinData)


# In[7]:


InsulinData.loc[:,'DateTime'] = pd.to_datetime(InsulinData.Date.astype(str)+' '+InsulinData.Time.astype(str))
InsulinData = InsulinData.drop(['Date', 'Time'], axis=1)
CGMData.loc[:,'DateTime'] = pd.to_datetime(CGMData.Date.astype(str)+' '+CGMData.Time.astype(str))
CGMData = CGMData.drop(['Date', 'Time'], axis=1)


# In[8]:


lastmealtime = pd.to_datetime('2000-01-01 00:00:00')
lastindex = 0
for index, row in InsulinData.iterrows():
    dif = row['DateTime'] - lastmealtime
    dif = dif.total_seconds()
    dif = dif/60
    if( dif <= 120 ):
        InsulinData = InsulinData.drop(lastindex)
    lastmealtime = row['DateTime']
    lastindex = index
# InsulinData.head(10)
# print(InsulinData)


# In[9]:


startofmeal = pd.DataFrame(columns=['Index'])
CGMDataCopy = CGMData
for index1, row1 in InsulinData.iterrows():
    for index2, row2 in CGMDataCopy.iterrows():
        if(row2['DateTime'] >= row1['DateTime']):
            new_row = pd.DataFrame([[index2]], columns=['Index'])
            startofmeal  = startofmeal.append(new_row)
            break
        CGMDataCopy = CGMDataCopy.drop(index2)


# In[10]:


new_row2 = []
for index1, row in startofmeal.iterrows():
    new_row1 = []
    for i in range(int(row['Index'])-6, int(row['Index'])+24):
        if(0 <= i <= 55343):
            new_row1.append(CGMData['Glucose'][i])
    new_row2.append(new_row1)
mealdata = pd.DataFrame(new_row2)
mealdata.to_csv('mealdata.csv', index=False , header= False)


# In[17]:


new_row3 = []
for index1, row in startofmeal.iterrows():
    new_row1 = []
    for i in range(int(row['Index'])+25, int(row['Index'])+49):
        if(0 <= i <= 55343):
            new_row1.append(CGMData['Glucose'][i])
    new_row3.append(new_row1)
nonmealData = pd.DataFrame(new_row3)
nonmealData.to_csv('nonmealData.csv', index=False , header= False)


# In[11]:


def peak(df):    
    peak_val =0
    loca = 0
    for index,z in df.iteritems():
        if(z > peak_val):
            peak_val = z
            loca = index
    return(peak_val, loca)


# In[12]:


mealdata = pd.read_csv('mealdata.csv', sep= ',', header = None,)

mealfeatureMatrix = pd.DataFrame(columns=['peak_val', 'thow', 'CGMn'])
for index, row in mealdata.iterrows():
    peak_val, thow = peak(row)
    meal_val = row[0]
    CGMn = (peak_val- meal_val)/meal_val
    df2 = pd.DataFrame([[peak_val, thow, CGMn]], columns=['peak_val', 'thow', 'CGMn'])
    mealfeatureMatrix  = mealfeatureMatrix.append(df2)
mealfeatureMatrix['label'] = 1
# print(mealfeatureMatrix)
# mealfeatureMatrix.head(30)


# In[18]:


nonmealdata = pd.read_csv('nonmealData.csv', sep= ',', header = None,)

nonmealfeatureMatrix = pd.DataFrame(columns=['peak_val', 'thow', 'CGMn'])
for index, row in nonmealdata.iterrows():
    peak_val, thow = peak(row)
    meal_val = row[0]
    CGMn = (peak_val- meal_val)/meal_val
    df2 = pd.DataFrame([[peak_val, thow, CGMn]], columns=['peak_val', 'thow', 'CGMn'])
    nonmealfeatureMatrix  = nonmealfeatureMatrix.append(df2)
nonmealfeatureMatrix['label'] = 0
# print(nonmealfeatureMatrix)
# nonmealfeatureMatrix.head(30)


# In[19]:


from sklearn.utils import shuffle
mealfeatureMatrix = shuffle(pd.concat([mealfeatureMatrix,nonmealfeatureMatrix],axis=0).fillna(0)).reset_index().drop(columns = ['index'])
# print(mealfeatureMatrix)


# In[20]:


feature_matrix = StandardScaler().fit_transform(mealfeatureMatrix.drop(columns= ['label']))
PrincipalComAna = PCA(n_components=3)
principalCom = PrincipalComAna.fit_transform(feature_matrix)
DF3 = pd.DataFrame(data = principalCom, columns = ['Atribute1', 'Atribute2', 'Atribute3'])


# In[21]:


Classifier = RandomForestClassifier()
X, y= DF3, mealfeatureMatrix['label']
Classifier.fit(X,y)
filename = 'model.pickle'
pickle.dump(Classifier, open(filename, 'wb'))

