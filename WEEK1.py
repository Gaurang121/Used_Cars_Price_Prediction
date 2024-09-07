#!/usr/bin/env python
# coding: utf-8

# # LAB 1

# # Data Acquisition
# <p>
# A data set is typically a file containing data stored in one of several formats. Common file formats containing data sets include: .csv, .json, .xlsx etc. The data set can be stored in different places, on your local machine, on a server or a websiite, cloud storage and so on.<br>
# 
# To analyse data in a Python notebook, we need to bring the data set into the notebook. In this section, you will learn how to load a data set into our Jupyter Notebook.<br>
# 
# In our case, the Automobile Data set is an online source, and it is in a CSV (comma separated value) format. Let's use this data set as an example to practice reading data.
# <ul>
#     <li>Data source: <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data" target="_blank">https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data</a></li>
#     <li>Data type: csv</li>
# </ul>
# The Pandas Library is a very popular and very useful tool that enables us to read various datasets into a data frame; our Jupyter notebook platforms have a built-in <b>Pandas Library</b> so that all we need to do is import Pandas without installing.
# </p>
# 

# In[8]:


import pandas as pd
import numpy as np


# In[9]:


#Reading the data


# In[10]:


pip install pyodide.http


# In[14]:


#copy the link below and paste it in the browser to download the dataset 
file_path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv'


# In[24]:


auto=pd.read_csv("auto.csv",header=None)
auto.head()


# In[20]:


#we could have directly used the URL as shown in the below example without downloading the dataset 


# In[29]:


exp=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv",header=None)


# In[30]:


exp.head(5)


# In[31]:


auto.tail(3)


# In[33]:


#In the above dataset we can see that we have no headers, so we go back to the code in line 24 and ass the following parameter :- header=None


# ## Adding Headers

# In[40]:


Headers=["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
for i in Headers:
    print(i)


# In[44]:


auto.columns=Headers
auto.columns


# In[45]:


auto.head(3)


# In[46]:


#Replacing the ? symbol with NaN symbol


# In[57]:


auto=auto.replace('?',np.NaN)
auto.head(3)


# In[60]:


auto.columns


# In[61]:


#SAVING THE DATASET


# In[62]:


auto.to_csv("automobile.csv")


# In[63]:


#Now the dataset automobile.csv is a pre-processed Dataset


# ###  SELF EVALUATION

# In[64]:


file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"


# In[65]:


#we use the df name to for dataframe in which we are going to store the dataset


# In[69]:


df=pd.read_csv(file_path,header=None)
df.head(5)


# In[68]:


#since we see that the dataset has no headers already we first import the dataset without headers by editng the line no. 67


# In[70]:


#now we need to add headers to the dataset


# In[74]:


headers=["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core", 
         "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg","Price"]

for x in headers:
    print(x)


# In[75]:


df.columns=headers


# In[80]:


df.head(5)


# In[83]:


df.replace('?',np.nan, inplace = True)
df.head(5)


# In[87]:


#to print the datatypes of the columns we use df.dtypes


# In[88]:


df.dtypes


# In[89]:


#Statistical Description of the dataset


# In[90]:


df.describe()


# In[93]:


#You can also use include='all' as an argument to get summaries for object-type columns.


# In[94]:


df.describe(include="all")


# In[99]:


#For summary information of the dataset
#Using the info() Method gives an overview of the top and bottom 30 rows of the DataFrame, useful for quick visual inspection.


# In[100]:


df.info()


# In[ ]:




