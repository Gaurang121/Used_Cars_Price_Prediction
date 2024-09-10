#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
import matplotlib.pylab as plt
import numpy as np


# In[3]:


#We are going to use the same dataset which was used in the previous lab and was saved as "automobile.csv".


# In[18]:


df=pd.read_csv("automobile.csv")
df.head(10)


#  <h4>Evaluating for Missing Data</h4>
# 
# The missing values are converted by default. Use the following functions to identify these missing values. You can use two methods to detect missing data:
# <ol>
#     <li><b>.isnull()</b></li>
#     <li><b>.notnull()</b></li>
# </ol>
# The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data.
# 

# In[7]:


missing_data=df.isnull()
missing_data.head(10)


# In[17]:


#Counting the number of missing values in each column
#using the for loop.


# In[9]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# Based on the summary above, each column has 205 rows of data and seven of the columns containing missing data:
# <ol>
#     <li>"normalized-losses": 41 missing data</li>
#     <li>"num-of-doors": 2 missing data</li>
#     <li>"bore": 4 missing data</li>
#     <li>"stroke" : 4 missing data</li>
#     <li>"horsepower": 2 missing data</li>
#     <li>"peak-rpm": 2 missing data</li>
#     <li>"price": 4 missing data</li>
# </ol>
# 

# ## Deal with missing data
# <b>How should you deal with missing data?</b>
# 
# <ol>
#     <li>Drop data<br>
#         a. Drop the whole row<br>
#         b. Drop the whole column
#     </li>
#     <li>Replace data<br>
#         a. Replace it by mean<br>
#         b. Replace it by frequency<br>
#         c. Replace it based on other functions
#     </li>
# </ol>
# 

# In[19]:


#It is easily identifiable that which data should be corrected through which technique just by looking at the data type.


# You should only drop whole columns if most entries in the column are empty. In the data set, none of the columns are empty enough to drop entirely.
# You have some freedom in choosing which method to replace data; however, some methods may seem more reasonable than others. Apply each method to different columns:
# 
# <b>Replace by mean:</b>
# <ul>
#     <li>"normalized-losses": 41 missing data, replace them with mean</li>
#     <li>"stroke": 4 missing data, replace them with mean</li>
#     <li>"bore": 4 missing data, replace them with mean</li>
#     <li>"horsepower": 2 missing data, replace them with mean</li>
#     <li>"peak-rpm": 2 missing data, replace them with mean</li>
# </ul>
# 
# <b>Replace by frequency:</b>
# <ul>
#     <li>"num-of-doors": 2 missing data, replace them with "four". 
#         <ul>
#             <li>Reason: 84% sedans are four doors. Since four doors is most frequent, it is most likely to occur</li>
#         </ul>
#     </li>
# </ul>
# 
# <b>Drop the whole row:</b>
# <ul>
#     <li>"price": 4 missing data, simply delete the whole row
#         <ul>
#             <li>Reason: You want to predict price. You cannot use any data entry without price data for prediction; therefore any row now without price data is not useful to you.</li>
#         </ul>
#     </li>
# </ul>
# 

# In[27]:


avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)


# In[34]:


#Replacing the "NaN" values with the mean value in Normalized-losses


# In[33]:


df["normalized-losses"].replace(np.nan, avg_norm_loss,inplace=True)
df["normalized-losses"].head(3)


# In[25]:


avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)


# In[35]:


#Replacing the "NaN" values with the mean value in Bore-column


# In[40]:


df["bore"].replace(np.nan,avg_bore,inplace=True)
df.head(3)


# In[41]:


avg_stroke_column=df["stroke"].astype("float").mean(axis=0)
print("Average of stroke column :",avg_stroke_column)


# In[42]:


#Repalcing the missing values of Stroke column with the mean value


# In[44]:


df["stroke"].replace(np.nan,avg_stroke_column,inplace=True)
df["stroke"].head(3)


# In[45]:


avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)


# In[46]:


#Replacing the missing values of the HorsePwer Column with the mean value


# In[48]:


df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
df['horsepower'].head(3)


# In[49]:


avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)


# In[50]:


#Replacing the missing values of the PeakRpm Column with the mean value


# In[53]:


df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
df["peak-rpm"].head(3)


# In[54]:


#To see which values are present in a particular column, we can use the ".value_counts()" method:


# In[55]:


df['num-of-doors'].value_counts()


# In[56]:


#You can see that four doors is the most common type. We can also use the ".idxmax()" method to calculate the most common type automatically:


# In[57]:


df['num-of-doors'].value_counts().idxmax()


# In[60]:


#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)
df["num-of-doors"].head(3)


# In[61]:


#Finally, drop all rows that do not have price data:


# In[62]:


df.dropna(subset=["price"],axis=0,inplace=True)


# In[63]:


# reset index, because we dropped two rows
df.reset_index(drop=True, inplace=True)


# In[64]:


df.head()


# In[65]:


#Now, we have a data set with no missing values.


# In[67]:


df.dtypes


# <p>As you can see above, some columns are not of the correct data type. Numerical variables should have type 'float' or 'int', and variables with strings such as categories should have type 'object'. For example, the numerical values 'bore' and 'stroke' describe the engines, so you should expect them to be of the type 'float' or 'int'; however, they are shown as type 'object'. You have to convert data types into a proper format for each column using the "astype()" method.</p> 
# 

# <h4>Convert data types to proper format</h4>

# In[68]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


# In[69]:


df.dtypes


# In[70]:


#Now you finally obtained the cleansed data set with no missing values and with all data in its proper format.


# ## Data Standardization
# <p>
# You usually collect data from different agencies in different formats.
# (Data standardization is also a term for a particular type of data normalization where you subtract the mean and divide by the standard deviation.)
# </p>
#     
# <b>What is standardization?</b>
# <p>Standardization is the process of transforming data into a common format, allowing the researcher to make the meaningful comparison.
# </p>
# 
# <b>Example</b>
# <p>Transform mpg to L/100km:</p>
# <p>In your data set, the fuel consumption columns "city-mpg" and "highway-mpg" are represented by mpg (miles per gallon) unit. Assume you are developing an application in a country that accepts the fuel consumption with L/100km standard.</p>
# <p>You will need to apply <b>data transformation</b> to transform mpg into L/100km.</p>
# 

# <p>Use this formula for unit conversion:<p>
# L/100km = 235 / mpg
# <p>You can do many mathematical operations directly using Pandas.</p>
# 

# In[71]:


df.head()


# In[74]:


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data 
df.head()


# In[73]:


# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)

# check your transformed data 
df.head()


# ## Data Normalization
# 
# <b>Why normalization?</b>
# <p>Normalization is the process of transforming values of several variables into a similar range. Typical normalizations include 
# <ol>
#     <li>scaling the variable so the variable average is 0</li>
#     <li>scaling the variable so the variance is 1</li> 
#     <li>scaling the variable so the variable values range from 0 to 1</li>
# </ol>
# </p>
# 
# <b>Example</b>
# <p>To demonstrate normalization, say you want to scale the columns "length", "width" and "height".</p>
# <p><b>Target:</b> normalize those variables so their value ranges from 0 to 1</p>
# <p><b>Approach:</b> replace the original value by (original value)/(maximum value)</p>
# 

# In[79]:


#We are going to be using the simple feature scaling technique of Normalization

# replace (original value) by (original value)/(maximum value)


# In[78]:


df["length"]=df["length"]/df["length"].max()
df["length"].head(3)


# In[80]:


df['width'] = df['width']/df['width'].max()
df["width"].head(3)


# In[81]:


df['height'] = df['height']/df['height'].max() 

# show the scaled columns
df[["length","width","height"]].head()


# In[82]:


#Here you've normalized "length", "width" and "height" to fall in the range of [0,1].


# ## Binning
# <b>Why binning?</b>
# <p>
#     Binning is a process of transforming continuous numerical variables into discrete categorical 'bins' for grouped analysis.
# </p>
# 
# <b>Example: </b>
# <p>In your data set, "horsepower" is a real valued variable ranging from 48 to 288 and it has 59 unique values. What if you only care about the price difference between cars with high horsepower, medium horsepower, and little horsepower (3 types)? You can rearrange them into three ‘bins' to simplify analysis.</p>
# 
# <p>Use the Pandas method 'cut' to segment the 'horsepower' column into 3 bins.</p>
# 

#  Convert data to correct format:
# 

# In[83]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# Plot the histogram of horsepower to see the distribution of horsepower.

# In[95]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# <p>Find 3 bins of equal size bandwidth by using Numpy's <code>linspace(start_value, end_value, numbers_generated</code> function.</p>
# <p>Since you want to include the minimum value of horsepower, set start_value = min(df["horsepower"]).</p>
# <p>Since you want to include the maximum value of horsepower, set end_value = max(df["horsepower"]).</p>
# <p>Since you are building 3 bins of equal length, you need 4 dividers, so numbers_generated = 4.</p>
# 

# Build a bin array with a minimum value to a maximum value by using the bandwidth calculated above. The values will determine when one bin ends and another begins.

# In[96]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# Set group  names:

# In[97]:


group_names = ['Low', 'Medium', 'High']


# Apply the function "cut" to determine what each value of `df['horsepower']` belongs to. 
# 

# In[98]:


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


# See the number of vehicles in each bin:

# In[99]:


df["horsepower-binned"].value_counts()


# Plot the distribution of each bin:

# In[100]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# <p>
#     Look at the data frame above carefully. You will find that the last column provides the bins for "horsepower" based on 3 categories ("Low", "Medium" and "High"). 
# </p>
# <p>
#     You successfully narrowed down the intervals from 59 to 3!
# </p>
# 

# <h3>Bins Visualization</h3>
# Normally, you use a histogram to visualize the distribution of bins we created above. 

# In[101]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot


# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# The plot above shows the binning result for the attribute "horsepower". 

# # Indicator Variable
# <b>What is an indicator variable?</b>
# <p>
#     An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning. 
# </p>
# 
# <b>Why use indicator variables?</b>
# <p>
#     You use indicator variables so you can use categorical variables for regression analysis in the later modules.
# </p>
# <b>Example</b>
# <p>
#     The column "fuel-type" has two unique values: "gas" or "diesel". Regression doesn't understand words, only numbers. To use this attribute in regression analysis, you can convert "fuel-type" to indicator variables.
# </p>
# 
# <p>
#     Use the Panda method 'get_dummies' to assign numerical values to different categories of fuel type. 
# </p>
# 

# In[102]:


df.columns


# Get the indicator
# variables and assign it to data frame "dummy_variable_1":
# 

# In[103]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# Change the column names for clarity:

# In[104]:


dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()


# In the data frame, column 'fuel-type' now has values for 'gas' and 'diesel' as 0s and 1s.

# In[105]:


# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)


# In[106]:


df.head()


# The last two columns are now the indicator variable representation of the fuel-type variable. They're all 0s and 1s now.

# In[107]:


# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()


# In[108]:


# merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)


# In[109]:


#Now we have completed the task of Data Wrangling and can save the file in the csv format


# In[110]:


df.to_csv('clean_df_auto.csv')


# In[ ]:




