import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift
from sklearn import metrics

# Load input data
X = np.loadtxt('data/data_clustering.txt', delimiter=',')

# Estimate the bandwith of X
bandwith_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Cluster data with MeanShift
meanshift_model = MeanShift(bandwidth=bandwith_X, bin_seeding=True)
meanshift_model.fit(X)

# Extra the centers of clusters
cluster_centers = meanshift_model.cluster_centers_
print('\n Centers of clusters: \n ', cluster_centers)

# Estimate the number of clusters
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print('\nNumber of clusters in input data = ', num_clusters)

# Plot the points and cluster centers
plt.figure()
markers = 'o*xvserf'
for i, marker in zip(range(num_clusters), markers):
    # Plot points that belong to the current cluster
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='red')

    # Plot the cluster center
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
             markerfacecolor='black', markeredgecolor='black',
             markersize=15)

plt.title('Clusters')
plt.show()
