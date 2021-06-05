#!/usr/bin/env python
# coding: utf-8

# In[1]:
from io import StringIO
import dvc.api
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
from dask import dataframe as dd 

# In[3]:
#input
with dvc.api.open(
        'data/test_data.csv',
        repo='https://github.com/shruthi-git-actions/dvc_v1.git',
        remote='remote_1',
        rev="experiment",
        encoding='utf-8'
        ) as fd2:
    main_test=pd.read_csv(fd2)
#data
with dvc.api.open(
        'data/column_list_new.csv',
        repo='https://github.com/shruthi-git-actions/dvc_v1.git',
        remote='remote_1',
        rev="experiment",
        encoding='utf-8'
        ) as fd3:
    columns_req=pd.read_csv(fd3)





main_dfff = dd.from_pandas(main_test, npartitions=7)


# In[4]:


df=main_dfff[columns_req["Variable_list"]]


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
print("hi")

# In[10]:

#model

clf = dvc.api.read(
    'HumanEvent_Model.pkl',
    repo='https://github.com/shruthi-git-actions/dvc_v1.git',
    mode='rb')

#with open(Pkl_Filename, 'rb') as file:  
#    clf = pickle.load(file)

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



