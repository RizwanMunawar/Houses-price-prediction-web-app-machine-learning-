#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('USA_Housing.csv')


# In[3]:


data.head(5)


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data.columns


# In[6]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[7]:


x = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)


# In[9]:


classifier = LinearRegression()


# In[10]:


classifier.fit(X_train, y_train)


# In[11]:


from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error


# In[12]:


predictions = classifier.predict(X_test)


# In[13]:


print(f'R^2 score: {r2_score(y_true=y_test, y_pred=predictions):.2f}')
print(f'MAE score: {mean_absolute_error(y_true=y_test, y_pred=predictions):.2f}')
print(f'EVS score: {explained_variance_score(y_true=y_test, y_pred=predictions):.2f}')
rp = sns.regplot(x=y_test, y=predictions)


# In[14]:


import pickle
with open('USA_Housing_Model.pkl', 'wb') as file:
    pickle.dump(classifier, file)

