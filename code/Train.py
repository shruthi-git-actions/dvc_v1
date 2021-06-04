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
import dvc.api
import csv


# In[ ]:

'''resource_url = dvc.api.get_url(
    'data/data.xml',
    repo='git@github.com:shruthi-git-actions/dvc_v1.git'
    )

print(resource_url)
'''

with dvc.api.open(
        'train_data.csv',
        repo='git@github.com:shruthi-git-actions/dvc_v1.git',
        remote='remote_storage'
        ) as fd:
    main_df=csv.reader(fd)
#url="https://drive.google.com/file/d/1mzsSx_8EPDCLQ77Md4aOg6V9D-769tGP/view?usp=sharing"
#path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
#main_df1 = pd.read_csv(main_df)
#main_test = dd.read_csv('test_data.csv')
#url1="https://drive.google.com/file/d/1Xn8q8UwNgNlpjfMiSxn-vV5TzYjDiYRN/view?usp=sharing"
#path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]
#columns_req = pd.read_csv(data/column_list_new.csv)




