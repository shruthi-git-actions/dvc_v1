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
import io

with dvc.api.open("HumanEvent_Model.pkl", mode="rb") as f:
  stream = io.BytesIO(f.read())
msg = pickle.load(stream)
print(msg)
'''
clf=dvc.api.read(
        'HumanEvent_Model.pkl',
        repo='https://github.com/shruthi-git-actions/dvc_v1.git',
        remote="remote_storage",
        mode='rb'
        )
'''    
print ("hi")