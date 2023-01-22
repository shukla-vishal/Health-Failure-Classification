#!/usr/bin/env python
# coding: utf-8

# ###### Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import plotly.express as px
import plotly.graph_objs as go


# ###### Loading the Dataset

# In[2]:


df=pd.read_csv("heart_failure_clinical_records_dataset.csv")


# In[3]:


df


# In[4]:


print('The Dataset has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))


# #### Numerical Features

# In[5]:


numCols = list(df.select_dtypes(exclude='object').columns)
print(f"There are {len(numCols)} numerical features:\n", numCols)


# #### Categoical Features

# In[6]:


catCols = list(df.select_dtypes(include='object').columns)
print(f"There are {len(catCols)} numerical features:\n", catCols)


# In[7]:


df.info()


# In[8]:


df.apply(lambda x: len(x.unique()))


# ###### Finding the Null Values

# In[9]:


df.isnull().sum()


# ###### Finding Duplicate Values

# In[10]:


df.duplicated().sum()


# ###### Applying One Hot Encoding

# In[11]:


df_oh= df

def one_hot_encoding(data,column):
    data= pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
    data= data.drop([column], axis=1)
    return data
cols= ['Married']

for col in cols:
    df_oh= one_hot_encoding(df_oh, col)


# In[12]:


df_oh


# ###### Dropping the Column

# In[13]:


dff= df_oh.drop(columns=['Married_No', 'Married_Yes'], axis=1)


# In[14]:


dff


# ###### Statistical Parameters of Dataset

# In[15]:


dff.describe().T.sort_values(ascending =0,by='mean').style.background_gradient(cmap='BuGn').bar(subset=['std'], color='red').bar(subset=['mean'], color='blue')


# In[16]:


dff.corr().style.background_gradient(cmap='gist_heat')


# ###  Data Visualisation

# ###### Gender
# 0 for Man
# 1 for Woman

# In[25]:


dff.sex.value_counts()


# In[26]:


gender_counts = dff.sex.value_counts()
plt.figure(figsize=(12, 6))
plt.pie(gender_counts, labels = ['0','1'], autopct ='%.1f%%', startangle = 90, explode = [0.1, 0], colors = ['lightskyblue', 'plum'])
plt.title("Gender Distribution (Male or Female)")


# ###### Number of patients with Anaemia
# 
# 0 = No
# 1 = Yes

# In[27]:


df.anaemia.value_counts()


# In[28]:


anaemia_counts =dff.anaemia.value_counts()
plt.figure(figsize=(12, 6))
plt.pie(anaemia_counts, labels =['0','1']  , autopct ='%.1f%%', startangle = 90, explode=[0.1, 0], colors =['lightgreen', 'red'])
plt.title("% of Patients have Anaemia")


# ######  Number of Patients with Diabetes
# 
# 0 - No
# 1 - Yes

# In[29]:


dff.diabetes.value_counts()


# In[30]:


diabetes_counts =dff.diabetes.value_counts()
plt.figure(figsize=(12, 6))
plt.pie(diabetes_counts, labels =['0','1']  , autopct ='%.1f%%', startangle = 90, explode=[0.1, 0], colors =['aquamarine', 'lightcoral'])
plt.title("% of Patients have Diabetes")


# ###### Number of Patients with High Blood Pressure
# 
# 0 - No
# 1 - Yes

# In[31]:


dff.high_blood_pressure.value_counts()


# In[32]:


blood_pressure_counts =dff.high_blood_pressure.value_counts()
plt.figure(figsize=(12, 6))
plt.pie(blood_pressure_counts, labels =['0','1']  , autopct ='%.1f%%', startangle = 90, explode=[0.1, 0], colors =['orange', 'brown'])
plt.title("% of Patients have High blood Pressure")


# ###### Number of Patients that has a Smoking Habit
# 
# 0 - No
# 1 - Yes

# In[33]:


dff.smoking.value_counts()


# In[34]:


smoking_counts =dff.smoking.value_counts()
plt.figure(figsize=(12, 6))
plt.pie(smoking_counts, labels =['Non-smoker','Smoker']  , autopct ='%.1f%%', startangle = 90, explode=[0.1, 0], colors =['blue', 'coral'])
plt.title("% of Patients have Smoking Habit")


# ###### Number of Patients Died During Follow up Period
# 
# 0 - No
# 1 - Yes

# In[36]:


dff.DEATH_EVENT.value_counts()


# In[37]:


dead_counts =dff.DEATH_EVENT.value_counts()
plt.figure(figsize=(12, 6))

plt.pie(dead_counts, labels =['0', '1']  , autopct ='%.1f%%', startangle = 90, explode=[0.1, 0], colors =['lightgreen', 'red'])
plt.title("% of Patients Dead & Alive")


# In[17]:


f, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 15))

sns.countplot(x='anaemia',data=dff,  ax=ax[0,0])

sns.countplot(x='diabetes',data=dff,  ax=ax[1,0])

sns.countplot(x='ejection_fraction',data=dff,  ax=ax[0,1])

sns.countplot(x='high_blood_pressure',data=dff,  ax=ax[1,1])

sns.countplot(x='sex',data=dff,  ax=ax[0,2])

sns.countplot(x='smoking',data=dff,  ax=ax[1,2])


# ###### Correlated Features

# In[18]:


correlated = dff.corr().DEATH_EVENT.sort_values(ascending=False)


# In[19]:


correlated


# In[20]:


sns.distplot(dff['age'])
plt.show()


# In[21]:


fig = px.box(dff, y="age", x="smoking", color="DEATH_EVENT", points="all",  hover_data=dff.columns)
fig.update_layout(title_text="Death depend on smoking")
fig.show()


# In[22]:


import warnings
warnings.filterwarnings("ignore")

true_anaemia = dff[dff["sex"]==1]
false_anaemia = dff[dff["sex"]==0]

true_anaemia_nodeath = true_anaemia[dff["DEATH_EVENT"]==0]
true_anaemia_death = true_anaemia[dff["DEATH_EVENT"]==1]
false_anaemia_nodeath = false_anaemia[dff["DEATH_EVENT"]==0]
false_anaemia_death = false_anaemia[dff["DEATH_EVENT"]==1]

labels = ['true_anaemia_nodeath','true_anaemia_death', 'false_anaemia_nodeath', 'false_anaemia_death']
values = [len(true_anaemia[dff["DEATH_EVENT"]==0]),len(true_anaemia[dff["DEATH_EVENT"]==1]),
         len(false_anaemia[df["DEATH_EVENT"]==0]),len(false_anaemia[dff["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Anaemia analysis")
fig.show()


# In[38]:


true_smoking = dff[dff["sex"]==1]
false_smoking = dff[dff["sex"]==0]

true_smoking_nodeath = true_smoking[dff["DEATH_EVENT"]==0]
true_smoking_death = true_smoking[dff["DEATH_EVENT"]==1]
false_smoking_nodeath = false_smoking[dff["DEATH_EVENT"]==0]
false_smoking_death = false_smoking[dff["DEATH_EVENT"]==1]

labels = ['true_smoking_nodeath','true_smoking_death', 'false_smoking_nodeath', 'false_smoking_death']
values = [len(true_smoking[dff["DEATH_EVENT"]==0]),len(true_smoking[dff["DEATH_EVENT"]==1]),
         len(false_smoking[dff["DEATH_EVENT"]==0]),len(false_smoking[dff["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="smoking analysis")
fig.show()


# ###### Applying Scaling

# In[40]:


cols_to_scale = ['age',  'creatinine_phosphokinase', 
       'ejection_fraction', 'platelets',
       'serum_creatinine', 'serum_sodium', 'time']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dff[cols_to_scale] = scaler.fit_transform(dff[cols_to_scale])


# In[41]:


dff


# In[42]:


X=dff.drop(['DEATH_EVENT'],axis='columns')
y=dff['DEATH_EVENT']


# In[43]:


X.shape


# In[44]:


y.shape


# ###### Splitting the Dataset

# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=5)


# In[46]:


X_train.shape

y_train.shape


# ###### Using Different Classification Models

# In[47]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


# In[48]:


model_params = {
             
    
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
            
        }
    } 
        
}


# ###### Applying Grid Search

# In[49]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
scores = []
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=cv, return_train_score=False)
    clf.fit(X,y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
dff= pd.DataFrame(scores,columns=['model','best_score','best_params'])
dff


# ### Support Vector Machine

# In[50]:


model=svm.SVC()
model.fit(X_train, y_train)


# In[51]:


model.score( X_test, y_test)


# In[52]:


y_predicted = model.predict(X_test)


# In[53]:


y_predicted[:5]


# In[54]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_predicted))


# In[55]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_predicted)
print(f'{mean_squared_error}: {mse}')


# ###### Confusion Matrix

# In[56]:


y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# ### Random Forest

# In[57]:


model=RandomForestClassifier()
model.fit(X_train, y_train)


# In[58]:


model.score( X_test, y_test)


# In[59]:


y_predicted = model.predict(X_test)


# In[60]:


y_predicted[:5]


# In[61]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_predicted))


# In[62]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_predicted)
print(f'{mean_squared_error}: {mse}')


# ###### Confusion Matrix

# In[63]:


y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

