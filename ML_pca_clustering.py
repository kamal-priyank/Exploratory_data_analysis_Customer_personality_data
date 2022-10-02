#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:07:32 2022

@author: group 2
"""


import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz as gp
from sklearn import decomposition as dcp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import preprocessing
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import time
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from matplotlib import colors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn


#%% Import data

customer_personality_unscaled = pd.read_csv('customer-personality.csv')

# convert dates to numbers
customer_personality_unscaled['Dt_Customer'] = [(datetime.strptime(x, '%d-%m-%Y')-datetime.strptime('01-01-1900', '%d-%m-%Y')).days for x in customer_personality_unscaled['Dt_Customer']]

# Create new features and delete some old
customer_personality_unscaled['is_single'] = [1 if i in ['Absurd', 'Alone', 'Divorced', 'Single', 'Widow', 'YOLO'] else 0 for i in customer_personality_unscaled['Marital_Status']]
customer_personality_unscaled["Total_Children"] = customer_personality_unscaled["Kidhome"] +customer_personality_unscaled["Teenhome"]
customer_personality_unscaled["is_parent"] = np.where(customer_personality_unscaled.Total_Children> 0, 1, 0)
customer_personality_unscaled["Total_Purchases"] = customer_personality_unscaled["NumDealsPurchases"] + customer_personality_unscaled["NumWebPurchases"] + customer_personality_unscaled["NumCatalogPurchases"] + customer_personality_unscaled["NumStorePurchases"]
customer_personality_unscaled["Total_Spent_Products"] = customer_personality_unscaled["MntWines"] + customer_personality_unscaled["MntFruits"] + customer_personality_unscaled["MntMeatProducts"] + customer_personality_unscaled["MntFishProducts"] + customer_personality_unscaled["MntSweetProducts"] + customer_personality_unscaled["MntGoldProds"]
customer_personality_unscaled["FamilySize"] = customer_personality_unscaled["Total_Children"] + customer_personality_unscaled["Marital_Status"].replace({"Married": 2, "Together": 2, "Absurd": 1, "Widow": 1, "YOLO": 1, "Divorced": 1, "Single": 1, "Alone": 1})

del customer_personality_unscaled['ID']

#%% Filter outoutliers (beyond 3 st deviations)

def filter_outliers(df, column_name):
    new_df = df[np.abs(df[column_name]-df[column_name].mean()) <= (3*df[column_name].std())].copy()
    return new_df

customer_personality_unscaled = filter_outliers(customer_personality_unscaled,'Income')
customer_personality_unscaled = filter_outliers(customer_personality_unscaled,'Year_Birth')
customer_personality_unscaled = filter_outliers(customer_personality_unscaled,'NumWebPurchases')
customer_personality_unscaled = filter_outliers(customer_personality_unscaled,'NumCatalogPurchases')

#%% Visualise correlation matrix

matrix = customer_personality_unscaled.corr()

fig, ax = plt.subplots(figsize=(10, 8))

mask = np.triu(np.ones_like(matrix, dtype=np.bool))
cmap1=sns.diverging_palette(250, 15, l=50, center="light", as_cmap=True)

mask = mask[1:, :-1]
corr = matrix.iloc[1:, :-1].copy()
sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", vmin=-1,
           vmax=1, cbar_kws={"shrink": .8}, annot_kws={"size": 7}, cmap=cmap1) #vlag, coolwarm
plt.yticks(rotation=0)
plt.show()

# fig.savefig("correlation_customer_personality.pdf")


#%% Get dummies and perform scaling

customer_personality_unscaled=customer_personality_unscaled.dropna()

customer_personality_unscaled=pd.get_dummies(customer_personality_unscaled)

scaler = StandardScaler()
scaler.fit(customer_personality_unscaled)
customer_personality=pd.DataFrame(scaler.transform(customer_personality_unscaled), columns=customer_personality_unscaled.columns)



#%% Optimal number ofcomponents in PCA

# Based on 50% explained variance rule of thumb we choose 5 components for this PCA analysis

explained_variance_ratio_cumul_sum=np.cumsum(PCA().fit(customer_personality).explained_variance_ratio_) #compute the cumulative sum

plt.title("Explained Variance Ratio by Component")
plt.plot(np.arange(1,customer_personality.shape[1]+1),explained_variance_ratio_cumul_sum) #so that the first component is at 1, not 0
plt.plot([1,30],[0.5,0.5])
plt.xlabel("Component")
plt.ylabel("Variance Ratio")
plt.show()


#%% PCA

pca = PCA(n_components = 5, random_state=643)
pca.fit(customer_personality)
customer_personality_pca = pd.DataFrame(pca.transform(customer_personality), columns = (["C"+str(i+1) for i in range((pca.transform(customer_personality)).shape[1])]))

def visualise_pca(fitted_pca, num_of_components):
    fig, ax = plt.subplots(figsize=(14, 8))
    loadings_sparse=pd.DataFrame(fitted_pca.components_[0:num_of_components,:].T).set_index(np.arange(1,customer_personality.shape[1]+1))
    
    loadings_sparse.columns = ["C"+str(i+1) for i in range(num_of_components)]
    loadings_sparse.index = customer_personality.columns
    ax=sns.heatmap(loadings_sparse, linewidth=0, cmap='PiYG', vmin=-1,vmax=1) #'PiYG'
    plt.show()
    # fig.savefig("PCA.pdf")
    
visualise_pca(pca,5)


#%% We use elbow method to find the optimal number of clusters

Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(customer_personality_pca)
Elbow_M.show()


#%% Use K-means to determine clusters and visualise

kmeans = KMeans(n_clusters=4, n_init=10, max_iter=300, random_state=244).fit(customer_personality_pca) #test_customers_segments, customer_personality
cmap2=sns.diverging_palette(15, 145, as_cmap=True)
ax=sns.heatmap(kmeans.cluster_centers_, linewidth=0, cmap=cmap2, xticklabels=customer_personality_pca.columns)
ax.set(xlabel='PCA components', ylabel='Clusters')
plt.show()

labels = kmeans.labels_

# print(sklearn.metrics.silhouette_score(customer_personality, labels))

#%% Visualise clusters for the each pair of the PCA components
def visualise_clusters(filename,customer_personality_data):
    # cmap2 = sns.color_palette("Paired",as_cmap=True)
    new_df=customer_personality_data.copy()
    new_df['Clusters']=labels
    sns.pairplot(new_df, hue='Clusters', palette='muted') #Paired, muted, rocket
    plt.show()
    # plt.savefig(filename+".pdf")
    
visualise_clusters("Components_vs_clusters",customer_personality_pca)


#%% Visualise clusters for selected variables

visualise_clusters("Variables_vs_clusters",customer_personality_unscaled[['Year_Birth','Income','FamilySize','Total_Spent_Products','Total_Purchases','is_parent','Dt_Customer']])


#%% All variables distribution for clusters

fig, axs = plt.subplots(6, 6, sharex=True, figsize=(40,25))

axs = axs.flatten()

for i in range(customer_personality_unscaled.shape[1]):
    sns.boxplot(ax=axs[i], x=labels, y=customer_personality_unscaled.iloc[:,i])
    axs[i].set_title(customer_personality_unscaled.iloc[:,i].name+' vs Clusters')
    # fig.suptitle('All variables distribution for clusters', fontsize=35)

    
plt.show()
# fig.savefig("Clusters.pdf")

#%% Top variables distribution for clusters

new_df_for_vis = customer_personality_unscaled[['Year_Birth','Income','FamilySize','Total_Spent_Products','Total_Purchases','is_parent','is_single','Dt_Customer']].copy()

fig, axs = plt.subplots(2, 4, sharex=True, figsize=(24,10))

axs = axs.flatten()

for i in range(new_df_for_vis.shape[1]):
    sns.boxplot(ax=axs[i], x=labels, y=new_df_for_vis.iloc[:,i])
    axs[i].set_title(new_df_for_vis.iloc[:,i].name+' vs Clusters')
    fig.suptitle('Top variables distribution for clusters', fontsize=35)
    
plt.show()
# fig.savefig("Clusters-top-variables.pdf")


