#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"


# In[3]:


df=pd.read_csv(file_path)


# In[4]:


df.head()


# In[5]:


#To Visualize variables we first need to see what kinf of Variable it is for selecting better Visualizing Techniques


# In[6]:


df.dtypes


# In[7]:


df["peak-rpm"].dtype


# In[8]:


df.corr() #This method did not work 


# In[9]:


df.corr(numeric_only=True) #Asked a friend


# In[10]:


#The correlation between the following columns: bore, stroke, compression-ratio, and horsepower.


# In[11]:


df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr(method='pearson')


# In[12]:


df[["body-style","peak-rpm"]].corr(numeric_only=True)


# <h2>Continuous Numerical Variables:</h2> 
# 
# <p>Continuous numerical variables are variables that may contain any value within some range. They can be of type "int64" or "float64". A great way to visualize these variables is by using scatterplots with fitted lines.</p>
# 
# <p>In order to start understanding the (linear) relationship between an individual variable and the price, we can use "regplot" which plots the scatterplot plus the fitted regression line for the data. This will be useful later on for visualizing the fit of the simple linear regression model as well. </p>
# 

# <h3>Positive Linear Relationship</h4>

# Let's find the scatterplot of "engine-size" and "price".
# 

# In[13]:


# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)


# As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables. Engine size seems like a pretty good predictor of price since the regression line is almost a perfect diagonal line.

# In[14]:


#We can even see the correlation between Price and Engine-Size


# In[15]:


df[["engine-size","price"]].corr()


# In[16]:


#We can clearly see that the Pearson correlation is 0.87 
#High correlation


# Highway mpg is a potential predictor variable of price. Let's find the scatterplot of "highway-mpg" and "price".

# In[17]:


sns.regplot(x="highway-mpg",y="price",data=df)


# As highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables. Highway mpg could potentially be a predictor of price.

# In[18]:


df[["highway-mpg","price"]].corr()


# In[19]:


#A negative correlation is being shown in the regression plot


# In[20]:


#There are many more variables which are affecting the price as we can see in the correlation table


# <h3>Weak Linear Relationship</h3>
# 

# Let's see if "peak-rpm" is a predictor variable of "price".

# In[21]:


sns.regplot(x="peak-rpm",y="price",data=df)


# Peak rpm does not seem like a good predictor of the price at all since the regression line is close to horizontal. Also, the data points are very scattered and far from the fitted line, showing lots of variability. Therefore, it's not a reliable variable.

# In[22]:


df[['peak-rpm','price']].corr()


# In[23]:


#The correlation between the two variabls is also not good


# In[24]:


df[["stroke","price"]].corr()


# In[25]:


sns.regplot(x="stroke",y="price",data=df)


# In[26]:


#There is a weak correlation between the variable 'stroke' and 'price.' as such regression will not work well. We can see this using "regplot" to demonstrate this.


# <h3>Categorical Variables</h3>
# 
# <p>These are variables that describe a 'characteristic' of a data unit, and are selected from a small group of categories. The categorical variables can have the type "object" or "int64". A good way to visualize categorical variables is by using boxplots.</p>
# 

# In[27]:


sns.boxplot(x="body-style",y="price",data=df)


# We see that the distributions of price between the different body-style categories have a significant overlap, so body-style would not be a good predictor of price. Let's examine engine "engine-location" and "price":

# In[28]:


sns.boxplot(x="engine-location",y="price",data=df)


# Here we see that the distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.

# Let's examine "drive-wheels" and "price".

# In[29]:


sns.boxplot(x="drive-wheels",y="price",data=df)


# Here we see that the distribution of price between the different drive-wheels categories differs. As such, drive-wheels could potentially be a predictor of price.

# ## Descriptive Statistical Analysis
# 

# <p>Let's first take a look at the variables by utilizing a description method.</p>
# 
# <p>The <b>describe</b> function automatically computes basic statistics for all continuous variables. Any NaN values are automatically skipped in these statistics.</p>
# 
# This will show:
# <ul>
#     <li>the count of that variable</li>
#     <li>the mean</li>
#     <li>the standard deviation (std)</li> 
#     <li>the minimum value</li>
#     <li>the IQR (Interquartile Range: 25%, 50% and 75%)</li>
#     <li>the maximum value</li>
# <ul>
# 

# In[30]:


df.describe()


# The default setting of "describe" skips variables of type object. We can apply the method "describe" on the variables of type 'object' as follows:

# In[31]:


df.describe(include="object")


# <h3>Value Counts</h3>

# <p>Value counts is a good way of understanding how many units of each characteristic/variable we have. We can apply the "value_counts" method on the column "drive-wheels". Don’t forget the method "value_counts" only works on pandas series, not pandas dataframes. As a result, we only include one bracket <code>df['drive-wheels']</code>, not two brackets <code>df[['drive-wheels']]</code>.</p>
# 

# In[32]:


df['drive-wheels'].value_counts()


# We can convert the series to a dataframe as follows:

# In[33]:


df['drive-wheels'].value_counts().to_frame()


# Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" and rename the column 'drive-wheels' to 'value_counts'.
# 
# 

# In[34]:


drive_wheel_counts=df['drive-wheels'].value_counts().to_frame()
drive_wheel_counts


# In[35]:


drive_wheel_counts.rename(columns={"drive-wheels":"Value_counts"},inplace=True)


# In[36]:


drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts


# We can repeat the above process for the variable 'engine-location'.

# In[37]:


# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head()


# After examining the value counts of the engine location, we see that engine location would not be a good predictor variable for the price. This is because we only have three cars with a rear engine and 198 with an engine in the front, so this result is skewed. Thus, we are not able to draw any conclusions about the engine location.

# ## Basics of Grouping

# <p>The "groupby" method groups data by different categories. The data is grouped based on one or several variables, and analysis is performed on the individual groups.</p>
# 
# <p>For example, let's group by the variable "drive-wheels". We see that there are 3 different categories of drive wheels.</p>
# 

# In[38]:


df["drive-wheels"].unique()


# <p>If we want to know, on average, which type of drive wheel is most valuable, we can group "drive-wheels" and then average them.</p>
# 
# <p>We can select the columns 'drive-wheels', 'body-style' and 'price', then assign it to the variable "df_group_one".</p>
# 

# In[39]:


df_group_one=df[["body-style","price","drive-wheels"]]
df_group_one


# We can then calculate the average price for each of the different categories of data.

# In[40]:


# grouping results
df_group_one = df_group_one.groupby(['drive-wheels']).mean()
df_group_one


# In[41]:


#Example of a group by since the above code is not working


# In[42]:


df_2 = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                              'Parrot', 'Parrot'],
                   'Max Speed': [380., 370., 24., 26.]})
df_2
df_2.groupby(['Animal']).mean()
        


# You can also group by multiple variables. For example, let's group by both 'drive-wheels' and 'body-style'. This groups the dataframe by the unique combination of 'drive-wheels' and 'body-style'. We can store the results in the variable 'grouped_test1'.

# In[43]:


df_gptest=df[["drive-wheels","price","body-style"]]
df_gptest


# In[44]:


grouped_test1=df_gptest.groupby(["drive-wheels","body-style"],as_index=False).mean()
grouped_test1


# This grouped data is much easier to visualize when it is made into a pivot table. A pivot table is like an Excel spreadsheet, with one variable along the column and another along the row. We can convert the dataframe to a pivot table using the method "pivot" to create a pivot table from the groups.
# 
# In this case, we will leave the drive-wheels variable as the rows of the table, and pivot body-style to become the columns of the table:

# In[45]:


grouped_pivot=grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot


# In[46]:


grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot


# In[47]:


#Use the "groupby" function to find the average "price" of each car based on "body-style".


# In[48]:


groupby_test2=df[['body-style','price']]
groupby_test2


# In[49]:


grouped_test_2=groupby_test2.groupby(['body-style'],as_index=False).mean()
grouped_test_2


# In[50]:


#Groupby completed


# <h4>Variables: Drive Wheels and Body Style vs. Price</h4>

# Let's use a heat map to visualize the relationship between Body Style vs Price.

# In[51]:


#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()


# The heatmap plots the target variable (price) proportional to colour with respect to the variables 'drive-wheel' and 'body-style' on the vertical and horizontal axis, respectively. This allows us to visualize how the price is related to 'drive-wheel' and 'body-style'.
# 
# The default labels convey no useful information to us. Let's change that:

# In[52]:


fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[53]:


#Visualization of data  ##REVISIT


# Visualization is very important in data science, and Python visualization packages provide great freedom. We will go more in-depth in a separate Python visualizations course.
# 
# The main question we want to answer in this module is, "What are the main characteristics which have the most impact on the car price?".
# 
# To get a better measure of the important characteristics, we look at the correlation of these variables with the car price. In other words: how is the car price dependent on this variable?

# ## Correlation and Causation
# 

# <p><b>Correlation</b>: a measure of the extent of interdependence between variables.</p>
# 
# <p><b>Causation</b>: the relationship between cause and effect between two variables.</p>
# 
# <p>It is important to know the difference between these two. Correlation does not imply causation. Determining correlation is much simpler  the determining causation as causation may require independent experimentation.</p>
# 

# <p><b>Pearson Correlation</b></p>
# <p>The Pearson Correlation measures the linear dependence between two variables X and Y.</p>
# <p>The resulting coefficient is a value between -1 and 1 inclusive, where:</p>
# <ul>
#     <li><b>1</b>: Perfect positive linear correlation.</li>
#     <li><b>0</b>: No linear correlation, the two variables most likely do not affect each other.</li>
#     <li><b>-1</b>: Perfect negative linear correlation.</li>
# </ul>
# 

# <p>Pearson Correlation is the default method of the function "corr". Like before, we can calculate the Pearson Correlation of the of the 'int64' or 'float64'  variables.</p>
# 

# In[54]:


df.corr(numeric_only=True)


# Sometimes we would like to know the significant of the correlation estimate.

# <b>P-value</b>
# <p>What is this P-value? The P-value is the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.</p>
# 
# By convention, when the
# <ul>
#     <li>p-value is $<$ 0.001: we say there is strong evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.05: there is moderate evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.1: there is weak evidence that the correlation is significant.</li>
#     <li>the p-value is $>$ 0.1: there is no evidence that the correlation is significant.</li>
# </ul>
# 

# In[56]:


#We can obtain this information using "stats" module in the "scipy" library.


# In[57]:


from scipy import stats


# <h3>Wheel-Base vs. Price</h3>

# Let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.
# 
# 

# In[59]:


pearson_coef,p_value=stats.pearsonr(df["wheel-base"],df["price"])
print("The pearson correlation is ",pearson_coef,"with a P-Value of ",p_value)


# <h4>Conclusion:</h4>
# <p>Since the p-value is $<$ 0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585).</p>
# 

# <h3>Horsepower vs. Price</h3>
# 

# In[63]:


pearson_coef,p_value=stats.pearsonr(df["horsepower"],df["price"])
print("Pearson correlation is :-",pearson_coef,"P-value is :-",p_value)


# <h4>conclusion:</h4>
# 
# since the P-value<0.001, the correlation between Horsepower and Price is statistically significant, and the linear relationship is strong (~0.809)

# <h3>Length vs. Price</h3>
# 
# Let's calculate the  Pearson Correlation Coefficient and P-value of 'length' and 'price'.
# 

# In[65]:


pearson_coef,p_value=stats.pearsonr(df["length"],df["price"])
print("pearson correlation is :- ",pearson_coef,"P-Value is :-",p_value)


# <h4>conclusion:</h4>
#     
# P-value < 0.001,the correlation between Lenght and Price is statistically significant and the linear relationship is moderately strong (~0.69)

# <h3>Width vs. Price</h3>
# 

# In[66]:


pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 


# <h4>conclusion:</h4>
#     
# since P-value<0.001 we can say that statistically there is a correlation between Width and Price and the Relationship is quite strong(~0.75)

# ### Curb-Weight vs. Price
# 

# In[69]:


pearson_coef,p_value=stats.pearsonr(df["curb-weight"],df["price"])
print("pearson correlation is:- ",pearson_coef,"P-value is:-",p_value)


# <h4>Conclusion:</h4>
#     
# since p_value<0.001 we can say that statistically there is a correlation between the variables Curb-weight and Price , also the Linear relationship between the variables is Strong (~0.83)

# <h3>Engine-Size vs. Price</h3>
# 
# Let's calculate the Pearson Correlation Coefficient and P-value of 'engine-size' and 'price':
# 

# In[70]:


pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# <h4>Conclusion:</h4>
# 
# <p>Since the p-value is $<$ 0.001, the correlation between engine-size and price is statistically significant, and the linear relationship is very strong (~0.872).</p>
# 

# <h3>Bore vs. Price</h3>

# In[73]:


pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value ) 


# <h4>Conclusion:</h4>
# <p>Since the p-value is $<$ 0.001, the correlation between bore and price is statistically significant, but the linear relationship is only moderate (~0.521).</p>
# 

#  We can relate the process for each 'city-mpg'  and 'highway-mpg':
# 

# <h3>City-mpg vs. Price</h3>

# In[74]:


pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# <h4>Conclusion:</h4>
# <p>Since the p-value is $<$ 0.001, the correlation between city-mpg and price is statistically significant, and the coefficient of about -0.687 shows that the relationship is negative and moderately strong.</p>
# 

# <h3>Highway-mpg vs. Price</h3>

# In[75]:


pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 


# #### Conclusion:
# Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant, and the coefficient of about -0.705 shows that the relationship is negative and moderately strong.

# <h3>Conclusion: Important Variables</h3>
# 

# <p>We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables:</p>
# 
# Continuous numerical variables:
# <ul>
#     <li>Length</li>
#     <li>Width</li>
#     <li>Curb-weight</li>
#     <li>Engine-size</li>
#     <li>Horsepower</li>
#     <li>City-mpg</li>
#     <li>Highway-mpg</li>
#     <li>Wheel-base</li>
#     <li>Bore</li>
# </ul>
#     
# Categorical variables:
# <ul>
#     <li>Drive-wheels</li>
# </ul>
# 
# <p>As we now move into building machine learning models to automate our analysis, feeding the model with variables that meaningfully affect our target variable will improve our model's prediction performance.</p>
# 

# ## END OF EDA 

# In[ ]:




