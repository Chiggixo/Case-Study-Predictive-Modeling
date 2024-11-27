#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[8]:


data = pd.read_csv('housing.csv')


# In[9]:


data.head()


# In[11]:


X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[14]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}, R² Score: {r2}")


# In[ ]:




