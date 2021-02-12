#!/usr/bin/env python
# coding: utf-8

# # Task-2: Prediction using unsupervised ml. From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.
# 
# # Presented by: Sahil Darji
# 

# - The dataset is available [here](https://drive.google.com/file/d/11Iq7YvbWZbt8VXjfm06brx66b10YiwK-/view)

# # Step 1- Importing all the required libraries and datasets

# In[1]:


# import libraries

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# import dataset

df = pd.read_csv('Iris.csv')
print('Read data successfully')


# In[19]:


# display data

df.head()


# # Finding out the optimum number of clusters for K Means

# In[21]:


# Finding the optimum number of clusters for k-means classification

x = df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# - Now it's clear that this method is called 'The Elbow Method'!!!
# 
# - Optimum clusters is where the elbow takes place.
# 
# - This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.

# # Applying Kmeans to the dataset

# In[22]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# # Visualising the clusters on first two columns & plotting the centroids of the clusters

# In[24]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'pink', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# # THANK YOU FOR WATCHING!
