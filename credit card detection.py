#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[22]:


# Example with a full file path
credit_card_data = pd.read_csv('creditcard.csv')


# In[23]:


credit_card_data.head()


# In[24]:


credit_card_data.tail()


# In[25]:


# dataset informations
credit_card_data.info()


# In[26]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[27]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[28]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[29]:


print(legit.shape)
print(fraud.shape)


# In[30]:


# statistical measures of the data
legit.Amount.describe()


# In[31]:


fraud.Amount.describe()


# In[32]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# In[33]:


legit_sample = legit.sample(n=492)


# In[34]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[35]:


new_dataset.head()


# In[36]:


new_dataset.tail()


# In[37]:


new_dataset['Class'].value_counts()


# In[38]:


new_dataset.groupby('Class').mean()


# In[39]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[40]:


print(X)


# In[41]:


print(Y)


# In[42]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[43]:


print(X.shape, X_train.shape, X_test.shape)


# In[44]:


model = LogisticRegression()


# In[45]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# In[46]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[47]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[ ]:




