#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[2]:


cd D:\FT\python\Ml\Assignment


# Part-1: data Exploration and Pre-processing

# In[3]:


# 1) Load the given dataset 
data=pd.read_csv("Python_Linear_Regres_a2.csv")


# In[4]:


# 2) Print all the column names 
data.columns


# In[5]:


# 3) Describe the data
data.describe()


# In[6]:


# 4) Drop address, date, postcode, YearBuilt, lattitude, longtitude columns
data.drop(['Address','Date','Postcode','YearBuilt','Lattitude','Longtitude'],axis=1,inplace=True)


# In[7]:


data.shape


# In[8]:


# 5) Find the count of null value in each column 
data.isnull().sum()


# In[9]:


# 6) Fill the null value of property count, distance, Bedroom2, Bathroom, Car with 0
for col in data.columns:
    if col=='Propertycount' or col=='Distance' or col=='Bedroom2' or col=='Bathroom' or col=='Car':
        data[col].fillna(0.0,inplace=True)
        
data.isnull().sum()


# In[10]:


# 7) Fill Null value of land size and bidding area columns with Mean

for col in data.columns:
    if col=='Landsize' or col=='BuildingArea':
        data[col].fillna(np.round(data[col].mean(),1),inplace=True)

data.isnull().sum()


# In[11]:


# 8) Find the unique value in method column 
data.Method.unique()


# In[12]:


data.dropna(inplace=True)


# In[13]:


# 9) Create a dummy data for categorical data one-hot encoding
data = pd.get_dummies(data, drop_first=True)


# In[14]:


data


# In[15]:


data.shape


# # Part-2: Working with Model 
# 

# In[16]:


# 1) Create the target data and feature data where target data is price 
x=data.drop('Price',axis=1)
y=data.Price


# In[17]:


# 2) Create a linear regression model for Target and feature data
# initialize the model
model=LinearRegression()
#make the tain and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
model.fit(x_train,y_train)


# In[19]:


get_ipython().system('pip install pickles')


# In[22]:


import pickle
with open('model.pkl','wb') as f:
    pickle.dump(model,f)


# In[ ]:


# 3) Check if the model is overfitting or underfitting or it is accurate 


# In[18]:


model.score(x_train,y_train)


# In[101]:


model.score(x_test,y_test)


# In[ ]:


# 4) If the model is overfitting then apply ridge and lasso regression algorithms 


# In[84]:


ridge_reg= Ridge(alpha=20, max_iter=100,)
ridge_reg.fit(x_train, y_train)


# In[103]:


lasso_reg = Lasso(alpha=50, max_iter=100)
lasso_reg.fit(x_train, y_train)


# In[104]:


lasso_reg.score(x_test, y_test)


# In[106]:


lasso_reg.score(x_train, y_train)


# In[107]:


# 5) Extract slope and intercept value from the model 
print(model.intercept_)
print(model.coef_)


# In[117]:


price_predicted=model.predict(x_train)
price_predicted


# In[118]:


# 6) Display Mean Squared Error 
mean_squared_error(y_test,price_predicted)


# In[119]:


# 7) Display Mean Absolute Error 
mean_absolute_error(y_test,price_predicted)


# In[123]:


# 8) Display Root mean Squared error 
np.sqrt(mean_squared_error(y_test,price_predicted))


# In[124]:


# 9) Display R2 score
r2_score(y_test,price_predicted)

