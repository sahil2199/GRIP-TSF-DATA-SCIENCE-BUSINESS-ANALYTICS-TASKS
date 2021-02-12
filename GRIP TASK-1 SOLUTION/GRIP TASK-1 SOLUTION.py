#!/usr/bin/env python
# coding: utf-8

# # Task-1: Predicting the scores of student on the basis of their study hours using simple linear regression
# 
# # Presented by: Sahil Darji
# 

# STEP 1 - Importing all the required Libraries and Datasets

# In[1]:


# Importing all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# Reading data from remote link

url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")

df.head(10)


# STEP 2 - Analyzing the Data for NULL values or Errors

# In[10]:


df.isnull().sum()


# No NULL value is found in the data. Therefore no Data Cleaning is required. Thus, we are good to go in Plotting the data.

# STEP 3 - Plotting the Data

# In[11]:


# Plotting the distribution of scores

df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# From the above graph we come to a conclusion that there is a positive linear relation between the number of hours studied and percentage of score. So we'll apply Linear Regression Model.
# 
# 

# STEP 4 - Preparing the Data for Training
# 

# In[12]:


# Preparing the Data for Training

X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values


# The division of data can also be done by running:
# 
# X = df[['Hours']].values
# 
# Y = df[['Scores']].values
# 
# The next step is to split this data into training and test sets.
# 
# We'll do this by using Scikit-Learn's built-in train_test_split() method as follows:

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)


# STEP 5 - Training the Algorithm

# Importing Linear Regression model and creating an instance

# In[14]:


# X = X.reshape((1,-1))

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train)

print("Training complete.")


# Plotting the Regression line

# In[15]:


line =regressor.coef_*X+regressor.intercept_
print(line)


# In[16]:


plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# STEP 6 - Making Predictions

# Now the Algorithm is trained and let's make some predictions.

# In[17]:


print(X_test)                               # Testing data - In Hours
y_pred = regressor.predict(X_test)          # Predicting the scores


# Comparing the Actual Data vs Predicted Results

# In[18]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})  
df


# STEP 7 - Predicting for 9.25 hrs/day

# In[19]:


hours = np.array(9.25).reshape(1, -1)
own_predt = regressor.predict(hours)
print('If the student reads for %0.3f hours then he will score %0.3f'%(9.25, own_predt[0]))


# STEP 8 - Evaluating the Model

# The final step is to evaluate the performance of algorithm with mean square error. This step is particularly important to compare how well different algorithms perform on a particular dataset.V

# In[20]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))


# THANK YOU FOR WATCHING!

# In[ ]:




