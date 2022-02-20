#this codes creates a set of benchmark tech company distribution for each year. The benchmark is then used
#to create the various tw measures.

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
from dask.dataframe import from_pandas


benchmark=pd.read_csv('/home/fm90b/aggdata/Benchkey.csv')
benchmark.columns = benchmark.columns.astype(int)
bench_all=pd.DataFrame()
for year in benchmark.columns:
        df_year= df_all.loc[df_all['jobdate'].dt.year==year]
        df_bench= df_year.loc[df_year.gvkey.isin(benchmark[year])]
        bench_all= bench_all.append(df_bench)
        print(year)
bench_all= bench_all.groupby('jobdate').mean().reset_index()
bench_all.drop(columns='gvkey', inplace= True)

bench_all.to_csv('benchall.csv')
