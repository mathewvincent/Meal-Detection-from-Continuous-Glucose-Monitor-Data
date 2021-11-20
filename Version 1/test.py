
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


def peak(df):    
    peak_val =0
    loca = 0
    for index,z in df.iteritems():
        if(z > peak_val):
            peak_val = z
            loca = index
    return(peak_val, loca)


# In[3]:


test = pd.read_csv('test.csv', sep= ',', header=None, index_col=None)

mealfeatureMatrix = pd.DataFrame(columns=['peak_val', 'thow', 'CGMn'])

for index, row in test.iterrows():
    peak_val, thow = peak(row)
    meal_val = row[0]
    CGMn = (peak_val- meal_val)/meal_val
    df2 = pd.DataFrame([[peak_val, thow, CGMn]], columns=['peak_val', 'thow', 'CGMn'])
    mealfeatureMatrix  = mealfeatureMatrix.append(df2)


# In[4]:


feature_matrix = StandardScaler().fit_transform(mealfeatureMatrix)
PrincipalComAna = PCA(n_components=3)
principalCom = PrincipalComAna.fit_transform(feature_matrix)
DF3 = pd.DataFrame(data = principalCom, columns = ['Atribute1', 'Atribute2', 'Atribute3'])


# In[5]:


filename = 'model.pickle'
loaded_model = pickle.load(open(filename, 'rb'))
prediction = loaded_model.predict(DF3)    
print (prediction)


# In[6]:


Result = pd.DataFrame(prediction)
Result.to_csv('Result.csv', index=False , header= False)

