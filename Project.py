#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

passmark = 40

df = pd.read_csv("StudentsPerformance.csv")

print(df.isnull().any())

p = sns.countplot(x="math score", data = df, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90)

df['Math_PassStatus'] = np.where(df['math score']<passmark, 'F', 'P')
df.Math_PassStatus.value_counts()

p = sns.countplot(x='parental level of education', data = df, hue='Math_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 

sns.countplot(x="reading score", data = df, palette="muted")
plt.show()

df['Reading_PassStatus'] = np.where(df['reading score']<passmark, 'F', 'P')
df.Reading_PassStatus.value_counts()

p = sns.countplot(x='parental level of education', data = df, hue='Reading_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 



p = sns.countplot(x="writing score", data = df, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90) 

df['Writing_PassStatus'] = np.where(df['writing score']<passmark, 'F', 'P')
df.Writing_PassStatus.value_counts()

p = sns.countplot(x='parental level of education', data = df, hue='Writing_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90)

df['OverAll_PassStatus'] = df.apply(lambda x : 'F' if x['Math_PassStatus'] == 'F' or 
                                    x['Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F' else 'P', axis =1)

df.OverAll_PassStatus.value_counts()

p = sns.countplot(x='parental level of education', data = df, hue='OverAll_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90)



df['Total_Marks'] = df['math score']+df['reading score']+df['writing score']
df['Percentage'] = df['Total_Marks']/3

p = sns.countplot(x="Percentage", data = df, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=0) 

def GetGrade(Percentage, OverAll_PassStatus):
    if ( OverAll_PassStatus == 'F'):
        return 'F'    
    if ( Percentage >= 80 ):
        return 'A'
    if ( Percentage >= 70):
        return 'B'
    if ( Percentage >= 60):
        return 'C'
    if ( Percentage >= 50):
        return 'D'
    if ( Percentage >= 40):
        return 'E'
    else: 
        return 'F'

df['Grade'] = df.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)

df.Grade.value_counts()

sns.countplot(x="Grade", data = df, order=['A','B','C','D','E','F'],  palette="muted")
plt.show()

p = sns.countplot(x='parental level of education', data = df, hue='Grade', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90)

ML_DataPoints = pd.read_csv("StudentsPerformance.csv", header=0, usecols=['math score', 'reading score', 'writing score'])
ML_Labels = pd.read_csv("StudentsPerformance.csv", header=0, usecols=['test preparation course'])

from sklearn.preprocessing import StandardScaler
MNScaler = StandardScaler()
MNScaler.fit(ML_DataPoints)
T_DataPoints = MNScaler.transform(ML_DataPoints)

from sklearn.preprocessing import LabelEncoder
LEncoder = LabelEncoder()
LEncoder.fit(ML_Labels) 
T_Labels = LEncoder.transform(ML_Labels)

from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(T_DataPoints, T_Labels, random_state = 10)


from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(
    n_estimators = 10,
    random_state = 3
)
RandomForest.fit(XTrain, YTrain)

ypred = RandomForest.predict(XTest)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(YTest,ypred)

import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(YTest, ypred)
roc_auc = metrics.auc(fpr, tpr)

print(cm)
print(roc_auc)

import pickle
pickle.dump(RandomForest,open('Project.pkl','wb'))
model=pickle.load(open('Project.pkl','rb'))









# In[ ]:




