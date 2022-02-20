import glob
import os
import pandas as pd   
import dask as dd 
df = dd.concat(map(pd.read_csv, glob.glob(os.path.join('', "agg*.csv"))))
dd.to_parquet('final_dask')
