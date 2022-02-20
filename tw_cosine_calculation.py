#this code calculates the rolling tw measure based on firm skill compositions
#As this process is computationally taxing, I would recommend using parallel processing for different chuncks of the data

from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
from dask.distributed import Client
client = Client()
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import pandas as pd
import pandas as pd
from zipfile import ZipFile
from io import StringIO
import fnmatch
# from pandas import ExcelWriter
# from dirty_cat import GapEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse
from sklearn.compose import ColumnTransformer
import dask
import dask.dataframe as dd
# from distributed import Client
# client = Client()
# import databricks.koalas as ks
# from pyspark.sql import SparkSession
import glob
import os
import pandas as pd   


beginging = np.datetime64("2010-01")
bench_all['jobdate']=pd.to_datetime(bench_all['jobdate'], format="%Y-%m").dt.to_period('M')
bench_all['start']=beginging
bench_all['start']=pd.to_datetime(bench_all['start'], format="%Y-%m").dt.to_period('M')
bench_all['t']=bench_all.jobdate.astype(int)  - bench_all.start.astype(int)
bench_all.drop(columns=['start','Unnamed: 0'] , inplace=True)

# df_all['jobdate']=pd.to_datetime(df_all['jobdate'], format="%Y-%m").dt.to_period('M')
df_all['start']=beginging
df_all['start']=pd.to_datetime(df_all['start'], format="%Y-%m").dt.to_period('M')
df_all['t']=df_all.jobdate.astype(int)  - df_all.start.astype(int)
df_all.drop(columns=['start'] , inplace=True)


bench_all=bench_all.groupby('jobdate').mean().reset_index()
bench_all.drop(columns= 'gvkey', inplace=True)


from scipy.spatial import distance
final=pd.DataFrame(columns=['t', 'gvkey', 'cosim'])


df_all=df_all[df_all.columns[~pd.Series(df_all.columns).str.startswith('skillcluster')]]
bench_all=bench_all[bench_all.columns[~pd.Series(bench_all.columns).str.startswith('skillcluster')]]


for time in range(1, 120):
    print(time)
    bench_twoyear= bench_all.loc[(bench_all['t']<=time-13) &(bench_all['t']>=time-24)]
    bench_twoyear= bench_twoyear.drop(columns=['t','jobdate', 'tot'])
    bench_sum= (bench_twoyear.sum()/12)
    shadow=df_all.loc[df_all['t']==time]
    for firm in shadow['gvkey']:
        timefirm=df_all.loc[(df_all['gvkey']==firm) & (df_all['t']<=time-1)& (df_all['t']>=time-12)]
        timefirm= timefirm.drop(columns=['t','jobdate','gvkey', 'tot' ])
        time_sum= (timefirm.sum()/12)
        cosim= (1- distance.cosine(bench_sum.values,time_sum.values))
        final = final.append({'t': time, 'gvkey': firm, 'cosim': cosim}, ignore_index=True)
import seaborn as sns
sns.displot(final, x="cosim")
final.cosim.describe()
final.to_csv('tw.csv')
