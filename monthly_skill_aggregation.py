# this work book looks to preprocess all the needed steps for the year 2010 as sample
#... to be continued for the following years
#This code only outputs the the skill distributions for the first month of 2010. similar code should be used for the
#rest of the data. It should be noted that this process is computationally and memory taxing. So, further paralelization can imporve the process.
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


with ZipFile('/home/fm90b/main_skill.zip') as zipfiles:
    file_list = zipfiles.namelist()

    #get only the csv files
#     dta_files = fnmatch.filter(file_list, "bskill_2010*")
#     for file_name in dta_files:
    df_skill2010 = pd.read_stata(zipfiles.open("bskill_2010.dta"))
    print(df_skill2010)
    
    
df_skillbyyear= pd.read_csv("skillbyyear.csv")
df_skillcusterby= pd.read_csv("skillclusterbyyear.csv")
df_skillcusterfamily= pd.read_csv("skillclusterfamilyname.csv")
df_skillbyyear.drop(df_skillbyyear.iloc[:, :12], inplace = True, axis = 1)
df_skillcusterby.drop(['Unnamed: 0'], axis = 1, inplace = True )
df_skillcusterfamily.drop(['Unnamed: 0'], axis = 1, inplace = True )
df_skillbyyear['skillname']= 'skill_' + df_skillbyyear['skillname'].astype(str)

df_skillcusterfamily['skillclusterfamilyname']= 'skillclusterfamily_' + df_skillcusterfamily['skillclusterfamilyname'].astype(str)
df_skillcusterby['skillclustername']= 'skillcluster_' + df_skillcusterby['skillclustername'].astype(str)


df_skill2010['skill']='skill_' + df_skill2010['skill'].astype(str)
df_skill2010['skillcluster']='skillcluster_' + df_skill2010['skillcluster'].astype(str)
df_skill2010['skillclusterfamily']='skillclusterfamily_' + df_skill2010['skillclusterfamily'].astype(str)




#dropping skills,clusters,family that have been cited less than a 100 times over the years - making sum column 

df_skillbyyear['tot'] = df_skillbyyear.sum(numeric_only=True, axis=1)
df_skillcusterby['tot'] = df_skillcusterby.sum(numeric_only=True, axis=1)
df_skillcusterfamily['tot'] = df_skillcusterfamily.sum(numeric_only=True, axis=1)




#dropping the rows below 100 aggregate
df_skillcusterfamily = df_skillcusterfamily[df_skillcusterfamily.tot > 100]
df_skillcusterby = df_skillcusterby[df_skillcusterby.tot > 100]
df_skillbyyear = df_skillbyyear[df_skillbyyear.tot > 100]




#reading the main file to assing the id/dates 
with ZipFile('/home/fm90b/main_skill.zip') as zipfiles:
    file_list = zipfiles.namelist()

    #get only the csv files
#     dta_files = fnmatch.filter(file_list, "bskill_2010*")
#     for file_name in dta_files:
    df_main2010 = pd.read_stata(zipfiles.open("b_2010.dta"))
    print(df_main2010)
    
columns = df_skillbyyear['skillname']
columns = columns.append(df_skillcusterby['skillclustername'])
columns=  columns.append(df_skillcusterfamily['skillclusterfamilyname'])



#creating the https://www.umassrc.org:444/node/c20b03/57818/notebooks/pre_jobscript-Copy1.ipynb#main df that would contain everything
# df_mother= pd.DataFrame(0, index=range(len(df_main2010['bgtjobid'])), columns= columns, dtype=np.int8 )
df_mother=pd.DataFrame()
df_mother['jobdate']=df_main2010['jobdate']
df_mother['gvkey']=df_main2010['gvkey']
df_mother['bgtjobid']=df_main2010['bgtjobid']
df_mother.set_index('bgtjobid', inplace= True)

#creating the column names to be onehot encoded 
del df_main2010
del df_skillcusterfamily
del df_skillcusterby
del df_skillbyyear

df_skill2010.drop(columns=['isspecialized', 'isbaseline', 'issoftware', 'salary'],  inplace = True )
df_skill2010["skill"] = df_skill2010["skill"].astype("category")  #remove the comment here for final submission 
df_skill2010["skillcluster"] = df_skill2010["skillcluster"].astype("category")   
df_skill2010["skillclusterfamily"] = df_skill2010["skillclusterfamily"].astype("category")

import dask
from dask import dataframe as dd 
from dask import delayed
from dask.dataframe import from_pandas

dd_mother2010 = from_pandas(df_mother, npartitions=12)
# dd_skill2010=from_pandas(df_skill2010, npartitions=12)

del df_mother
# del df_skill2010
dd_mother2010=client.persist(dd_mother2010)
# dd_skill2010= client.persist(dd_skill2010)


dd_mother2010['jobdate']= dd.to_datetime(dd_mother2010.jobdate, format="%Y-%m-%d")
dd_mother2010['jobdate'] =dd_mother2010['jobdate'].dt.to_period('M')
dd_mother2010['jobdate'].head()
dd_mother2010= dd_mother2010[(dd_mother2010['jobdate'] >= '2010-1') & (dd_mother2010['jobdate'] <= '2010-1')]
df_skill2010=df_skill2010.loc[df_skill2010.bgtjobid.isin(dd_mother2010.index)]
df_skill2010.reset_index(inplace=True)



#one hot encoding skill distributions here

# categorical_features = ['skill', 'skillcluster', 'skillclusterfamily']
categorical_transformer = OneHotEncoder()

categorical_transformer = OneHotEncoder()

coder=  OneHotEncoder(dtype=np.int8)
coded=coder.fit_transform(df_skill2010.select_dtypes('category')).toarray()
col= coder.categories_
col = np.concatenate( col, axis=0 )
df_coded= pd.DataFrame(data= coded, columns= col, dtype= np.int8)
df_coded['bgtjobid']=df_skill2010['bgtjobid']
del coded
del df_skill2010
dd_coded= from_pandas(df_coded, npartitions=12)
dd_coded=client.persist(dd_coded)
del df_coded

#aggrigating the data on single bgtid values 

dd_coded= dd_coded.groupby('bgtjobid').sum().compute()

# Removing adding the needed columns to regularize the format
coded_columns= dd_coded.columns
columns=pd.Series(columns)
# print(type(columns))
columns= pd.Index(columns.values)# this is to make the format the same between daks index and columns series (pandas)
col_toadd= columns.difference(coded_columns)
print(col_toadd)
col_toremove=coded_columns.difference(columns)
# col_toremove=col_toremove[2:] ------- add this line if the the mother columns are already added and thus removed 
dd_coded=dd_coded.drop(col_toremove, axis=1)
dd_mother2010['bgtjobid']=dd_mother2010.index
dd_mother2010.drop_duplicates(subset='bgtjobid', keep='first', inplace=True, ignore_index=False)
dd_coded=dd_coded.merge(dd_mother2010, left_index=True, right_index= True, validate= 'one_to_one')

# adding the columns needed to standardize with OG columns
for c in col_toadd:
    dd_coded[c]= 0
    print(c)
   
#groupby date and gvkey to get the final output
dd_coded['tot']=1 
# dd_coded['jobdate']= dd.to_datetime(dd_coded.jobdate, format="%Y-%m-%d")
# dd_coded['jobdate'] =dd_coded['jobdate'].dt.to_period('M')
group= ['jobdate','gvkey']
dd_coded= dd_coded.groupby(group).sum().reset_index() #maybe remove the compute if it bugs out


dd_coded.to_csv('agg2010_1.csv')


