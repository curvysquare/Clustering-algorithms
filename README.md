## Description:

### K-means 
- Coded from scratch the K-means algorithm: a representative based, clustering algorithm that uses centroid’s to creates K number of clusters, within a unsupervised clustering problems. The target function is to minimise the squared distance between data objects and the centroid in its clusters.
- This functions through the centroid’s being initialised as a random point within the feature- space, but not necessarily an actual datapoint. Next, each data object is assigned to its nearest centroid, all data objects with the same centroid form a cluster. The centroid is then updated according to the mean of the cluster. This occurs iterativley until convergence where the updating of a centroid does not cause any changes to the objects within each cluster, or the maximum number of iterations is reached 

### K-means++
- coded from scratch the improved K-means algorithm: The K-means++ algorithm improves upon K-means by selecting the initial centroid represen- tatives in a systematic way that increases the chances of optimal clustering. The first centroid is an actual datapoint from the dataset chosen at random, then the squared distance from this centroid to every other data point is calculated.
-  The squared distances are summed up and is used to create a probability distribution such that the next centroid is chosen with probability proportional to its squared distance from the centroid. This repeats for the next centroid, with the adjustment that the distance to the nearest centroid is used in constructing the probability distribution.
-The use of the probability distribution to choose the centroids ensures that the clusters will be more spaced out than that of random initialisation. After K number of centroids have been initialized, the process continues as per the standard K-means algorithm.

### Bi-secting K-means:
- coded from scrach the Bisecting k-means algorithm is a heirarchical divisive clustering algorithm that forms clusters in a ’top-down’ approach starting with a single cluster containing the entirety of the dataset.
- The cluster with the largest sum of squared distances between the objects of the cluster and the cluster centroid is subject to division into two new clusters. This target cluster is split using the k-means algorithm with the k being equalt to 2.
- This iterative procedure continues until the specified number of clusters is reached.

### Evaluation and comparison result 
- The effectivness of the clustering algorithms are compared using the silhouette coefficient
- Overall it can be concluded that the best clustering is achieved for equally for all algorithms with two clusters due to the country objects being easily separable.
- Yet, making the assumption that the ground truth clustering consists of four clusters; Kmeans ++ achieves the highest Silhouette Coefficent.
