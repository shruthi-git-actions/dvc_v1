#!/usr/bin/env python
# coding: utf-8

# In[ ]:
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


# In[ ]:
url="https://drive.google.com/file/d/1mzsSx_8EPDCLQ77Md4aOg6V9D-769tGP/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
main_df = pd.read_csv(path)
#main_test = dd.read_csv('test_data.csv')
url1="https://drive.google.com/file/d/1Xn8q8UwNgNlpjfMiSxn-vV5TzYjDiYRN/view?usp=sharing"
path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]
columns_req = pd.read_csv(path1)




# In[ ]:

main_dff = dd.from_pandas(main_df, npartitions=7)
df=main_dff[columns_req["Variable_list"]]
#df_test=main_test[columns_req["Variable_list"]]
no_columns=len(columns_req.index)


# In[ ]:


x=df.iloc[:,0:no_columns-1]
y_train=df.iloc[:,-1]
#test_x=df_test.iloc[:,0:no_columns-1]
#test_train=df_test.iloc[:,-1]


# In[ ]:


x=x.categorize()


# In[ ]:


de = DummyEncoder()
X_train = de.fit_transform(x)


# In[ ]:


client = Client(processes=False)             # create local cluster

clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
with joblib.parallel_backend("dask", scatter=[X_train, y_train]):
 clf.fit(X_train, y_train)


# In[ ]:



Pkl_Filename = "HumanEvent_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)


# In[ ]:


pred_train=clf.predict(X_train)


# In[ ]:


pred_train_df=dd.from_array(pred_train)
pred_train_df=pred_train_df.to_frame()
training=x.merge(pred_train_df)


# In[ ]:




training.to_csv("out/training_new.csv", single_file = True)


# In[ ]:
train_score=accuracy_score(np.array(y_train),np.array(pred_train))

print(train_score)
print ("the end ")




