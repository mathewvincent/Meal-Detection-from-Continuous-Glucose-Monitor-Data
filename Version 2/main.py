
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import math
import pickle
import warnings
warnings.filterwarnings("ignore")
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
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from datetime import datetime, date, time
from scipy import stats
from scipy.stats import pearsonr


# In[3]:


InsulinData = pd.read_csv("InsulinData.csv", header=None, sep=",")
InsulinData = InsulinData.drop(0)
InsulinData = InsulinData[[1,2,24]]
InsulinData.columns =['Date', 'Time', 'Meal']
InsulinData = InsulinData.reindex(index=InsulinData.index[::-1])
# InsulinData


# In[4]:


CGMData = pd.read_csv("CGMData.csv", sep=",", header = None)
CGMData = CGMData.drop(0)
CGMData = CGMData[[1,2,30]]
CGMData.columns =['Date', 'Time', 'Glucose']
CGMData = CGMData.reindex(index=CGMData.index[::-1])
# print(CGMData)


# In[ ]:


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


# In[ ]:


InsulinData.loc[:,'DateTime'] = pd.to_datetime(InsulinData.Date.astype(str)+' '+InsulinData.Time.astype(str))
InsulinData = InsulinData.drop(['Date', 'Time'], axis=1)
CGMData.loc[:,'DateTime'] = pd.to_datetime(CGMData.Date.astype(str)+' '+CGMData.Time.astype(str))
CGMData = CGMData.drop(['Date', 'Time'], axis=1)


# In[ ]:


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


# In[ ]:


mincarb = 100
maxcarb = 0
for index, row in InsulinData.iterrows():
    if(int(row['Meal']) < mincarb):
        mincarb = row['Meal']
    if(int(row['Meal']) > maxcarb):
        maxcarb = row['Meal']
print(mincarb, maxcarb)


# In[ ]:


binmatrix = pd.DataFrame(columns=['Bin'])
for index, row in InsulinData.iterrows():
    new_row = pd.DataFrame([[0]], columns=['Bin'])
    if(mincarb <= int(row['Meal']) <= mincarb+20):
        new_row = pd.DataFrame([[0]], columns=['Bin'])
    if(mincarb+20 < int(row['Meal']) <= mincarb+40):
        new_row = pd.DataFrame([[1]], columns=['Bin'])
    if(mincarb+40 < int(row['Meal']) <= mincarb+60):
        new_row = pd.DataFrame([[2]], columns=['Bin'])
    if(mincarb+60 < int(row['Meal']) <= mincarb+80):
        new_row = pd.DataFrame([[3]], columns=['Bin'])
    if(mincarb+80 < int(row['Meal']) <= mincarb+100):
        new_row = pd.DataFrame([[4]], columns=['Bin'])
    if(mincarb+80 < int(row['Meal']) <= maxcarb):
        new_row = pd.DataFrame([[5]], columns=['Bin'])
    binmatrix  = binmatrix.append(new_row)


# In[ ]:


startofmeal = pd.DataFrame(columns=['Index'])
CGMDataCopy = CGMData
for index1, row1 in InsulinData.iterrows():
    for index2, row2 in CGMDataCopy.iterrows():
        if(row2['DateTime'] >= row1['DateTime']):
            new_row = pd.DataFrame([[index2]], columns=['Index'])
            startofmeal  = startofmeal.append(new_row)
            break
        CGMDataCopy = CGMDataCopy.drop(index2)


# In[ ]:


new_row2 = []
for index1, row in startofmeal.iterrows():
    new_row1 = []
    for i in range(int(row['Index'])-6, int(row['Index'])+24):
        if(0 <= i <= 55343):
            new_row1.append(CGMData['Glucose'][i])
    new_row2.append(new_row1)
mealdata = pd.DataFrame(new_row2)
mealdata.to_csv('mealdata.csv', index=False , header= False)


# In[ ]:


def peak(df):    
    peak_val =0
    loca = 0
    for index,z in df.iteritems():
        if(z > peak_val):
            peak_val = z
            loca = index
    return(peak_val, loca)


# In[ ]:


mealdata = pd.read_csv('mealdata.csv', sep= ',', header = None,)

mealfeatureMatrix = pd.DataFrame(columns=['peak_val', 'thow', 'CGMn'])
for index, row in mealdata.iterrows():
    peak_val, thow = peak(row)
    meal_val = row[0]
    CGMn = (peak_val- meal_val)/meal_val
    df2 = pd.DataFrame([[peak_val, thow, CGMn]], columns=['peak_val', 'thow', 'CGMn'])
    mealfeatureMatrix  = mealfeatureMatrix.append(df2)
# print(mealfeatureMatrix)
# mealfeatureMatrix.head(30)


# # K-Mean

# In[ ]:


kmeancluster = KMeans(n_clusters=6)
kmean_predicted = kmeancluster.fit_predict(mealfeatureMatrix[['peak_val','thow']])
kmean_sse = kmeancluster.inertia_
kmean_predicted = pd.DataFrame(kmean_predicted, columns= ['predicted'] )


# In[ ]:


kmean_Entropy  = 0
kmean_Purity  = 0


# In[ ]:


# matrix
kmean_matrix = pd.DataFrame(columns=['predicted', 'actual'])
for index1, row in kmean_predicted.iterrows():
    for index2, row2 in binmatrix.iterrows():
        if(index1==index2):
            new_row = pd.DataFrame([[row['predicted'], row2['Bin']]], columns=['predicted', 'actual'])
            kmean_matrix  = kmean_matrix.append(new_row)
            break;
kmean_matrix     


# # DBSCAN

# In[ ]:


dbscanclustering = DBSCAN(eps =2 , min_samples = 11).fit(mealfeatureMatrix[['peak_val','thow']])
dbscan_predicted = dbscanclustering.labels_
len(set(dbscan_predicted))
dbscan_sse = kmeancluster.inertia_


# In[ ]:


dbscan_Entropy  = 0
dbscan_Purity  = 0


# In[ ]:


dbscan_predicted = pd.DataFrame(dbscan_predicted, columns= ['predicted'] )

dbscan_matrix = pd.DataFrame(columns=['predicted', 'actual'])
for index1, row in dbscan_predicted.iterrows():
    for index2, row2 in binmatrix.iterrows():
        if(index1==index2):
            new_row = pd.DataFrame([[row['predicted'], row2['Bin']]], columns=['predicted', 'actual'])
            dbscan_matrix  = dbscan_matrix.append(new_row)
            break;
dbscan_matrix   


# In[ ]:


Result = pd.DataFrame([[kmean_sse,dbscan_sse,kmean_Entropy,dbscan_Entropy,kmean_Purity,dbscan_Purity]])
Result.to_csv('Result.csv', index=False , header= False)

