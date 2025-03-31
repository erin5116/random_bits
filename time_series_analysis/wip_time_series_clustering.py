# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:36:10 2024

To investigate the sktime library for clustering of time series data

@author: ywan3672
"""

import SimpleITK as sitk
import os
import glob
import numpy as np
import pylab as plt

# 
from sklearn.model_selection import train_test_split

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm

from sktime.datasets import load_arrow_head

X, y = load_arrow_head(return_X_y=True)

temp = X.iloc[200]['dim_0'].values
print(temp.shape)
# X is a dataframe with 211 entries
# each entry is an 1D array (seems to be all 251 x 1) with values along the "time"
# y is a 211 x 1 array with strings "0",'1",'2" -> labels?

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#%% K-means
k_means = TimeSeriesKMeans(
    n_clusters=3,  # Number of desired centers
    init_algorithm="forgy",  # Center initialisation technique
    max_iter=10,  # Maximum number of iterations for refinement on training set
    metric="dtw",  # Distance metric to use
    averaging_method="mean",  # Averaging technique to use
    random_state=1,
)

k_means.fit(X_train)
plot_cluster_algorithm(k_means, X_test, k_means.n_clusters)
print('intertia = {0}'.format(k_means.inertia_))

y_pred = k_means.predict(X_test)
y_pred_prob = k_means.predict_proba(X_test)
print(y_test)

#%% K-menoids

k_medoids = TimeSeriesKMedoids(
    n_clusters=5,  # Number of desired centers
    init_algorithm="forgy",  # Center initialisation technique
    max_iter=10,  # Maximum number of iterations for refinement on training set
    verbose=False,  # Verbose
    metric="dtw",  # Distance metric to use
    random_state=1,
)

#%% Exploring use of intertia as metric to select number of cluster (K-means example)

for num_cluster in range(1,6):
    
    k_means = TimeSeriesKMeans(
        n_clusters=num_cluster,  # Number of desired centers
        init_algorithm="forgy",  # Center initialisation technique
        max_iter=10,  # Maximum number of iterations for refinement on training set
        metric="dtw",  # Distance metric to use
        averaging_method="mean",  # Averaging technique to use
        random_state=1,
    )
    
    k_means.fit(X_train)
    # plot_cluster_algorithm(k_means, X_test, k_means.n_clusters)
    
    # check the attributes of the TimeSeriesKMeans object
    # print('number of time series = {0}'.format(len(k_means.labels_)))
    print('number of cluster = {0}'.format(len(set(k_means.labels_))))
    # print('dimension of cluster center data = {0}'.format(k_means.cluster_centers_.shape))
    print('intertia = {0}'.format(k_means.inertia_)) # inertia = sum of squared distances of samples to their closest cluster center
    # print('number of iterations ran for = {0}\n'.format(k_means.n_iter_))

k_medoids.fit(X_train)
plot_cluster_algorithm(k_medoids, X_test, k_medoids.n_clusters)
print('intertia = {0}'.format(k_medoids.inertia_))
