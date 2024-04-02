#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


# In[2]:


import pandas as pd
dataset = pd.read_csv(r"C:\Users\kumar\Downloads\Oasis infobyte internship\Customer segmentation analysis\Mall_Customers.csv")
dataset.head(120)


# In[4]:


print(dataset.info())
print(dataset.describe())


# In[5]:


df = pd.DataFrame(dataset)
gender_counts = df['Genre'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(gender_counts.index, gender_counts.values, color=['red', 'brown'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
plt.show()


# In[6]:


X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []  
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[8]:


optimal_num_clusters = 5 
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
dataset['Cluster'] = cluster_labels
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='x', s=200, label='Cluster Centers')
plt.xlabel('Standardized Annual Income (k$)')
plt.ylabel('Standardized Spending Score (1-100)')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
print("Cluster Centers:")
print(scaler.inverse_transform(kmeans.cluster_centers_))


# In[12]:


clustered_data = dataset[['CustomerID', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']]
print(clustered_data.tail(-10))


# In[10]:


silhouette_avg = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")


# In[11]:


db_index = davies_bouldin_score(X, cluster_labels)
print(f"Davides-Bouldin Index: {db_index}")


# In[ ]:




