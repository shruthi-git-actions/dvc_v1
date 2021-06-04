#!/usr/bin/env python
# coding: utf-8

# In[1]:
from io import StringIO
import requests
import aiohttp
import dask_ml
import pandas as pd
import dask.dataframe as dd
from dask_ml.preprocessing import DummyEncoder
import numpy as np
from dask.distributed import Client
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle
import dask
from dask_ml.metrics import accuracy_score
import os, uuid
import json


# In[3]:


url_test="https://drive.google.com/file/d/1cd__STfaGNMOT5tKDKkXlLNcRqQypVoZ/view?usp=sharing"
path_test = 'https://drive.google.com/uc?export=download&id='+url_test.split('/')[-2]
main_df = pd.read_csv(path_test)

url1="https://drive.google.com/file/d/1Xn8q8UwNgNlpjfMiSxn-vV5TzYjDiYRN/view?usp=sharing"
path2 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]
columns_req = pd.read_csv(path2)

from dask import dataframe as dd 
main_dff = dd.from_pandas(main_df, npartitions=7)


# In[4]:


df=main_dff[columns_req["Variable_list"]]


# In[5]:


no_columns=len(columns_req.index)


# In[6]:





# In[7]:


x=df.iloc[:,0:no_columns-1]


# In[8]:


x=x.categorize()


# In[9]:


de = DummyEncoder()
X_test = de.fit_transform(x)


# In[10]:



Pkl_Filename = "HumanEvent_Model.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    clf = pickle.load(file)


# In[11]:


pred_test=clf.predict(X_test)


# In[12]:


pred_test_df=dd.from_array(pred_test)


# In[13]:


pred_test_df=pred_test_df.to_frame()


# In[14]:


prediction=x.merge(pred_test_df)


# In[16]:





# In[17]:



prediction.to_csv("out1/prediction_new.csv", single_file = True)



