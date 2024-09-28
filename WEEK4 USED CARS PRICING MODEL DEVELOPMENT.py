#!/usr/bin/env python
# coding: utf-8

# <p> Here we will develop several models that will predict the price of the car using the variables or features. This is just an estimate but should give us an objective idea of how much the car should cost.</p>

# Some questions we want to ask :-
# <ul>
#     <li>Do I know if the dealer is offering fair value for my trade-in?</li>
#     <li>Do I know if I put a fair value on my car?</li>
# </ul>
# <p>In data analytics, we often use <b>Model Development</b> to help us predict future observations from the data we have.</p>
# 
# <p>A model will help us understand the exact relationship between different variables and how these variables are used to predict the result.</p>
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"


# In[3]:


df=pd.read_csv(file_path)


# In[4]:


df.head()


# In[5]:


#The dataset we have is already pre processed so we need not do the preprocessing 


# In[8]:


sns.regplot(x="highway-mpg",y="price",data=df)


# <h2>1. Linear Regression and Multiple Linear Regression</h2>

# <h4>Linear Regression</h4>
# 

# <p>One example of a Data  Model that we will be using is:</p>
# <b>Simple Linear Regression</b>
# 
# <br>
# <p>Simple Linear Regression is a method to help us understand the relationship between two variables:</p>
# <ul>
#     <li>The predictor/independent variable (X)</li>
#     <li>The response/dependent variable (that we want to predict)(Y)</li>
# </ul>
# 
# <p>The result of Linear Regression is a <b>linear function</b> that predicts the response (dependent) variable as a function of the predictor (independent) variable.</p>
# 

# $$
#  Y: Response \ Variable\\\\\\
#  X: Predictor \ Variables
# $$

#  <b>Linear Function</b>
# $$
# Yhat = a + b  X
# $$
# 

# <ul>
#     <li>a refers to the <b>intercept</b> of the regression line, in other words: the value of Y when X is 0</li>
#     <li>b refers to the <b>slope</b> of the regression line, in other words: the value with which Y changes when X increases by 1 unit</li>
# </ul>
# 

# <h4>Create the linear regression object:</h4>

# In[5]:


from sklearn.linear_model import LinearRegression


# In[9]:


lm = LinearRegression()
lm


# <h4>How could "highway-mpg" help us predict car price?</h4>

# Using simple linear regression, we will create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable.

# In[10]:


X=df[["highway-mpg"]]
Y=df["price"]


# Fit the linear model using highway-mpg:

# In[12]:


lm.fit(X,Y)


#  We can output a prediction:

# In[13]:


Yhat=lm.predict(X)


# In[12]:


Yhat[0:5]


# <h4>We can also find the intercept and slope of the equation</h4>

# In[14]:


print("The intercept is:-",lm.intercept_)


# In[15]:


print("The slope is :- ", lm.coef_)


# <h3>What is the final estimated linear model we get?</h3>
# 

# As we saw above, we should get a final linear model with the structure:
# 

# $$
# Yhat = a + b  X
# $$

# <b>Price</b> = 38423.31 - 821.73 x <b>highway-mpg</b>

# In[15]:


#We were able to write the exact Linear equation with the help of slope and intercept


# <h4>Linear Regression model "lm1"</h4>

# In[16]:


lm1=LinearRegression()


# In[17]:


x=df[["engine-size"]]
y=df["price"]


# In[18]:


lm1.fit(x,y)


# In[19]:


var=lm1.predict(x)


# In[20]:


var[0:5]


# In[21]:


print("The Slope is:-",lm1.coef_,"and the Intercept is:- ",lm1.intercept_)


# ### Equation of the predicted line is:- 
# 
# using X and Y  
# Yhat=-7963.34 + 166.86*X
# 
# Price=-7963.34 + 166.86*df['engine-size']

# <h4>Multiple Linear Regression</h4>

# <p>What if we want to predict car price using more than one variable?</p>
# 
# <p>If we want to use more variables in our model to predict car price, we can use <b>Multiple Linear Regression</b>.
# Multiple Linear Regression is very similar to Simple Linear Regression, but this method is used to explain the relationship between one continuous response (dependent) variable and <b>two or more</b> predictor (independent) variables.
# Most of the real-world regression models involve multiple predictors. We will illustrate the structure by using four predictor variables, but these results can generalize to any integer:</p>
# 

# $$
# Yhat = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
# $$

# <p>From the previous section  we know that other good predictors of price could be:</p>
# <ul>
#     <li>Horsepower</li>
#     <li>Curb-weight</li>
#     <li>Engine-size</li>
#     <li>Highway-mpg</li>
# </ul>
# Let's develop a model using these variables as the predictor variables.
# 

# In[22]:


Z=df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y=df["price"]


# In[23]:


lm2=LinearRegression()


# In[24]:


lm2.fit(Z,y)


# In[25]:


var2=lm2.predict(Z)


# In[26]:


var2[0:5]


# In[27]:


print("The intercept is:- ",lm2.intercept_,"The slope is :-",lm2.coef_)


# As we saw above, we should get a final linear function with the structure:
# 
# $$
# Yhat = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
# $$
# 
# What is the linear function we get in this example?
# 

# <b>Price</b> = -15678.742628061467 + 52.65851272 x <b>horsepower</b> + 4.69878948 x <b>curb-weight</b> + 81.95906216 x <b>engine-size</b> + 33.58258185 x <b>highway-mpg</b>
# 

# In[51]:


#Another example of mulitple linear regression is  :-


# In[28]:


lm3=LinearRegression()


# In[29]:


lm3.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])


# In[58]:


lm3.intercept_,lm3.coef_


# <h2> Model Evaluation Using Visualization</h2>

# Now that we've developed some models, how do we evaluate our models and choose the best one? One way to do this is by using a visualization.

# <p>When it comes to simple linear regression, an excellent way to visualize the fit of our model is by using <b>regression plots</b>.</p>
# 
# <p>This plot will show a combination of a scattered data points (a <b>scatterplot</b>), as well as the fitted <b>linear regression</b> line going through the data. This will give us a reasonable estimate of the relationship between the two variables, the strength of the correlation, as well as the direction (positive or negative correlation).</p>
# 

# In[30]:


sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)


# <p>We can see from this plot that price is negatively correlated to highway-mpg since the regression slope is negative.
# 
# One thing to keep in mind when looking at a regression plot is to pay attention to how scattered the data points are around the regression line. This will give you a good indication of the variance of the data and whether a linear model would be the best fit or not. If the data is too far off from the line, this linear model might not be the best model for this data.
# 
# Let's compare this plot to the regression plot of "peak-rpm".</p>
# 

# In[31]:


sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)


# <p>Comparing the regression plot of "peak-rpm" and "highway-mpg", we see that the points for "highway-mpg" are much closer to the generated line and, on average, decrease. The points for "peak-rpm" have more spread around the predicted line and it is much harder to determine if the points are decreasing or increasing as the "peak-rpm" increases.</p>
# 

# <h4>Given the regression plots above, is "peak-rpm" or "highway-mpg" more strongly correlated with "price"? Use the method ".corr()" to verify your answer.</h4>

# In[32]:


#short data = sd


# In[33]:


sd=df[["highway-mpg","peak-rpm","price"]]
sd.head(3)


# In[34]:


sd.corr()


# <h3>Residual Plot</h3>
# 
# <p>A good way to visualize the variance of the data is to use a residual plot.</p>
# 
# <p>What is a <b>residual</b>?</p>
# 
# <p>The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e). When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.</p>
# 
# <p>So what is a <b>residual plot</b>?</p>
# 
# <p>A residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.</p>
# 
# <p>What do we pay attention to when looking at a residual plot?</p>
# 
# <p>We look at the spread of the residuals:</p>
# 
# <p>- If the points in a residual plot are <b>randomly spread out around the x-axis</b>, then a <b>linear model is appropriate</b> for the data.
# 
# Why is that? Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data.</p>
# 

# In[35]:


width = 6
height = 4
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.show()


# What is this plot telling us?
# 
# We can see from this residual plot that the residuals are not randomly spread around the x-axis, leading us to believe that maybe a non-linear model is more appropriate for this data.

# <h3>Multiple Linear Regression</h3>

# How do we visualize a model for Multiple Linear Regression? This gets a bit more complicated because you can't visualize it with regression or residual plot.
# 
# One way to look at the fit of the model is by looking at the distribution plot. We can look at the distribution of the fitted values that result from the model and compare it to the distribution of the actual values.

# In[36]:


Y_hat = lm2.predict(Z)


# In[37]:


plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# We can see that the fitted values are reasonably close to the actual values since the two distributions overlap a bit. However, there is definitely some room for improvement.

# In[29]:


#Understand the above code


# <h2>Polynomial Regression and Pipelines</h2>

# <p><b>Polynomial regression</b> is a particular case of the general linear regression model or multiple linear regression models.</p> 
# <p>We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.</p>
# 
# <p>There are different orders of polynomial regression:</p>
# 

# <center><b>Quadratic - 2nd Order</b></center>
# $$
# Yhat = a + b_1 X +b_2 X^2 
# $$
# 
# 
# <center><b>Cubic - 3rd Order</b></center>
# $$
# Yhat = a + b_1 X +b_2 X^2 +b_3 X^3\\\\\\
# $$
# 
# 
# <center><b>Higher-Order</b>:</center>
# $$
# Y = a + b_1 X +b_2 X^2 +b_3 X^3 ....\\
# $$
# 

# <p>We saw earlier that a linear model did not provide the best fit while using "highway-mpg" as the predictor variable. Let's see if we can try fitting a polynomial model to the data instead.</p>
# 

# In[70]:


#plotting the data


# In[39]:


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
#This is a function that needs to be called for visualization


# In[40]:


x = df['highway-mpg']
y = df['price']


# Let's fit the polynomial using the function <b>polyfit</b>, then use the function <b>poly1d</b> to display the polynomial function.
# 

# In[41]:


#Here we use polynomial of 3rd order
f=np.polyfit(x,y,3)
p=np.poly1d(f)
print(p)


# In[42]:


PlotPolly(p, x, y, 'highway-mpg')


# In[43]:


np.polyfit(x,y,3)


# <p>We can already see from plotting that this polynomial model performs better than the linear model. This is because the generated polynomial function  "hits" more of the data points.</p>

# <h4>Create 11 order polynomial model with the variables x and y from above.</h4>

# In[44]:


f1=np.polyfit(x,y,11)
p1=np.poly1d(f1)
print(p1)


# In[45]:


PlotPolly(p1,x,y,"highway_mpg")


# <p>The analytical expression for Multivariate Polynomial function gets complicated. For example, the expression for a second-order (degree=2) polynomial with two variables is given by:</p>
# 

# $$
# Yhat = a + b_1 X_1 +b_2 X_2 +b_3 X_1 X_2+b_4 X_1^2+b_5 X_2^2
# $$
# 

# We can perform a polynomial transform on multiple features. First, we import the module:

# In[46]:


from sklearn.preprocessing import PolynomialFeatures


# In[47]:


pr=PolynomialFeatures(degree=2)
pr


# In[48]:


Z_pr=pr.fit_transform(Z)


# In[49]:


Z.shape


# In[50]:


Z_pr.shape


# <h2>Pipeline</h2>
# 

# <p>Data Pipelines simplify the steps of processing the data. We use the module <b>Pipeline</b> to create a pipeline. We also use <b>StandardScaler</b> as a step in our pipeline.</p>

# In[51]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# We create the pipeline by creating a list of tuples including the name of the model or estimator and its corresponding constructor.
# 

# In[52]:


Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]


# We input the list as an argument to the pipeline constructor:
# 

# In[53]:


pipe=Pipeline(Input)
pipe


# First, we convert the data type Z to type float to avoid conversion warnings that may appear as a result of StandardScaler taking float inputs.
# 
# Then, we can normalize the data,  perform a transform and fit the model simultaneously. 
# 

# In[54]:


Z = Z.astype(float)
pipe.fit(Z,y)


#  Similarly,  we can normalize the data, perform a transform and produce a prediction  simultaneously.
# 

# In[55]:


ypipe=pipe.predict(Z)
ypipe[0:4]


# In[56]:


#Create a pipeline that standardizes the data, then produce a prediction using a linear regression model using the features Z and target y.


# In[57]:


Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]


# <h2> Measures for In-Sample Evaluation</h2>
# 

# <p>When evaluating our models, not only do we want to visualize the results, but we also want a quantitative measure to determine how accurate the model is.</p>
# 
# <p>Two very important measures that are often used in Statistics to determine the accuracy of a model are:</p>
# <ul>
#     <li><b>R^2 / R-squared</b></li>
#     <li><b>Mean Squared Error (MSE)</b></li>
# </ul>
#     
# <b>R-squared</b>
# 
# <p>R squared, also known as the coefficient of determination, is a measure to indicate how close the data is to the fitted regression line.</p>
#     
# <p>The value of the R-squared is the percentage of variation of the response variable (y) that is explained by a linear model.</p>
# 
# 
# 
# <b>Mean Squared Error (MSE)</b>
# 
# <p>The Mean Squared Error measures the average of the squares of errors. That is, the difference between actual value (y) and the estimated value (ŷ).</p>
# 

# <h3>Model 1: Simple Linear Regression</h3>
# 

# In[6]:


#Let's calculate R^2


# In[59]:


#highway_mpg_fit
lm.fit(X,Y)


# In[62]:


print("The R^2 is :-",lm.score(X,Y))


# We can say that ~49.659% of the variation of the price is explained by this simple linear model "horsepower_fit".

# In[63]:


#Calculating MSE


# We can predict the output i.e., "yhat" using the predict method, where X is the input variable:
# 

# In[64]:


yhat=lm.predict(X)


# In[65]:


print("The output of the first 4 variables is :- ",yhat[0:4])


# <h4>Let's import the function <b>mean_squared_error</b> from the module <b>metrics</b>:</h4>

# In[67]:


from sklearn.metrics import mean_squared_error


# In[68]:


#we can compare the actual values with the predicted values to check te accuracy of the model


# In[69]:


MSE=mean_squared_error(df["price"],yhat)
print("The Mean Squared Error of predicted and actual price is :- ",MSE)


# <h3>Model 2: Multiple Linear Regression</h3>

# In[70]:


#Calculating R^2


# In[73]:


lm.fit(Z,df["price"])
print("The R-square is:- ",lm.score(Z,df["price"]))


# We can say that ~80.896 % of the variation of price is explained by this multiple linear regression "multi_fit".

# In[74]:


#Calculating MSE


# In[75]:


Y_predict_multifit = lm.predict(Z)


# In[76]:


print("The first 4 predicted values are:- ",Y_predict_multifit[0:4])


# In[77]:


#we compare the predicted results with the actual results:
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))


# <h3>Model 3: Polynomial Fit</h3>

# In[78]:


#Calculating R^2


# Let’s import the function <b>r2_score</b> from the module <b>metrics</b> as we are using a different function.

# In[79]:


from sklearn.metrics import r2_score


# In[81]:


#Now we can apply this function to get the value of R-Square

r_squared=r2_score(y,p(x))
print("The R-square is:- ",r_squared)


# In[82]:


#Calculating MSE


# In[83]:


MSE3=mean_squared_error(df["price"],p(x))
print("The MSE is :- ",MSE3)


# <h2>Prediction and Decision Making</h2>
# <h3>Prediction</h3>
# 
# <p>In the previous section, we trained the model using the method <b>fit</b>. Now we will use the method <b>predict</b> to produce a prediction. Lets import <b>pyplot</b> for plotting; we will also be using some functions from numpy.</p>
# 

# Create a new input:

# In[84]:


new_input=np.arange(1, 100, 1).reshape(-1, 1)


# In[85]:


#fit the model


# In[86]:


lm.fit(X,Y)
lm


# In[87]:


#Producing a prediction


# In[88]:


yhat=lm.predict(new_input)
yhat[0:5]


# In[89]:


plt.plot(new_input, yhat)
plt.show()


# <h3>Decision Making: Determining a Good Model Fit</h3>

# <p>Now that we have visualized the different models, and generated the R-squared and MSE values for the fits, how do we determine a good model fit?
# <ul>
#     <li><i>What is a good R-squared value?</i></li>
# </ul>
# </p>
# 
# <p>When comparing models, <b>the model with the higher R-squared value is a better fit</b> for the data.
# <ul>
#     <li><i>What is a good MSE?</i></li>
# </ul>
# </p>
# 
# <p>When comparing models, <b>the model with the smallest MSE value is a better fit</b> for the data.</p>
# 
# 
# <h4>Let's take a look at the values for the different models.</h4>
# <p>Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price.
# <ul>
#     <li>R-squared: 0.49659118843391759</li>
#     <li>MSE: 3.16 x10^7</li>
# </ul>
# </p>
#     
# <p>Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.
# <ul>
#     <li>R-squared: 0.80896354913783497</li>
#     <li>MSE: 1.2 x10^7</li>
# </ul>
# </p>
#     
# <p>Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.
# <ul>
#     <li>R-squared: 0.6741946663906514</li>
#     <li>MSE: 2.05 x 10^7</li>
# </ul>
# </p>
# 

# <h3>Simple Linear Regression Model (SLR) vs Multiple Linear Regression Model (MLR)</h3>
# 

# <p>Usually, the more variables you have, the better your model is at predicting, but this is not always true. Sometimes you may not have enough data, you may run into numerical problems, or many of the variables may not be useful and even act as noise. As a result, you should always check the MSE and R^2.</p>
# 
# <p>In order to compare the results of the MLR vs SLR models, we look at a combination of both the R-squared and MSE to make the best conclusion about the fit of the model.
# <ul>
#     <li><b>MSE</b>: The MSE of SLR is  3.16x10^7  while MLR has an MSE of 1.2 x10^7.  The MSE of MLR is much smaller.</li>
#     <li><b>R-squared</b>: In this case, we can also see that there is a big difference between the R-squared of the SLR and the R-squared of the MLR. The R-squared for the SLR (~0.497) is very small compared to the R-squared for the MLR (~0.809).</li>
# </ul>
# </p>
# 
# This R-squared in combination with the MSE show that MLR seems like the better model fit in this case compared to SLR.
# 

# <h3>Simple Linear Model (SLR) vs. Polynomial Fit</h3>

# <ul>
#     <li><b>MSE</b>: We can see that Polynomial Fit brought down the MSE, since this MSE is smaller than the one from the SLR.</li> 
#     <li><b>R-squared</b>: The R-squared for the Polynomial Fit is larger than the R-squared for the SLR, so the Polynomial Fit also brought up the R-squared quite a bit.</li>
# </ul>
# <p>Since the Polynomial Fit resulted in a lower MSE and a higher R-squared, we can conclude that this was a better fit model than the simple linear regression for predicting "price" with "highway-mpg" as a predictor variable.</p>
# 

# <h3>Multiple Linear Regression (MLR) vs. Polynomial Fit</h3>
# 

# <ul>
#     <li><b>MSE</b>: The MSE for the MLR is smaller than the MSE for the Polynomial Fit.</li>
#     <li><b>R-squared</b>: The R-squared for the MLR is also much larger than for the Polynomial Fit.</li>
# </ul>

# <h3>CONCLUSION</h3>

# In[ ]:




