import sys
import os
from time import time
%matplotlib inline
from urllib.request import urlopen
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import math
  
def load_mnist():
    images_url = 'https://github.com/guptashvm/Data/blob/master/data/train-images-idx3-ubyte?raw=true'
    with urlopen(images_url) as urlopened:
      fd = urlopened.read()
      loaded = np.frombuffer(fd,dtype=np.uint8)
      trX = loaded[16:].reshape((60000,28*28)).astype(float)
 
    labels_url = 'https://github.com/guptashvm/Data/blob/master/data/train-labels-idx1-ubyte?raw=true'
    with urlopen(labels_url) as urlopened:
      fd = urlopened.read()
      loaded = np.frombuffer(fd,dtype=np.uint8)
      trY = loaded[8:].reshape((60000))
 
    trY = np.asarray(trY)
 
    X = trX / 255.
    y = trY
 
    subset  = [i for i, t in enumerate(y) if t in [1, 0, 2, 3]]
    X, y = X.astype('float32')[subset], y[subset]
    return X[:1000], y[:1000]
  
# Run the following code to load the data we need. Here, **X** is the array of
# images, **y** is the array of labels, and **X2d** is the array of projections of 
# the images into two-dimensional space using PCA. The two-dimensional scatterplot of 
# the top two principal components of the images is displayed, where the color of 
# each point represents its label. 

X, y = load_mnist()
pca = PCA(n_components=2)
pca.fit(X)
X2d = X.dot(pca.components_.T)

def plot_with_colors(Xs, ys):
  for i, _ in enumerate(ys):
    if ys[i] == 0:
      plt.plot([Xs[i, 0]], [Xs[i, 1]], 'r.')
    elif ys[i] == 1:
      plt.plot([Xs[i, 0]], [Xs[i, 1]], 'b.')
    elif ys[i] == 2:
      plt.plot([Xs[i, 0]], [Xs[i, 1]], 'g.')
    elif ys[i] == 3:
      plt.plot([Xs[i, 0]], [Xs[i, 1]], 'y.')
  plt.show()

plot_with_colors(X2d, y)
  
# (a) Implement the standard k-means algorithm. Please complete the function
# __kmeans__ defined below. You are NOT allowed to use any existing code of
# **kmeans** for this problem.


def kmeans(X, k = 4, max_iter = 500, random_state=0):
#   Inputs:
#       X: input data matrix, numpy array with shape (n * d), n: number of data points, d: feature dimension
#       k: number of clusters
#       max_iters: maximum iterations
#   Output:
#       clustering label for each data point

    assert len(X) > k, 'illegal inputs'
    np.random.seed(random_state)

  # randomly select k data points as centers
    idx = np.random.choice(len(X), k, replace=False)
    centers = X[idx]
    new_centers = np.zeros((k, X.shape[1]))

  # please complete the following code: 

    labels = np.zeros(len(X))

    from scipy.spatial import distance
    for i in range(max_iter):
        H = distance.cdist(X, centers, 'euclidean')
        
    
        for j in range(len(X)):
            labels[j] = np.argsort(H[j])[0]
    
        for j in range(k):
            new_centers[j] = X[labels==j].mean(axis=0)
            
        if np.sum(new_centers == centers) == k:
            break
        centers = new_centers
        
    return labels
  
  
print("Results of standard k-means algorithm using the 2D dataset")
plot_with_colors(X2d, kmeans(X2d))

print("Results of standard k-means algorithm using the original dataset")
plot_with_colors(X2d, kmeans(X))

# (b) Run your **kmeans** function on the dataset (of the top two PCA components given by array X2d). Set the number of clusters to 4. Visualize the result by coloring the 2D points in (a) according to their **clustering labels** returned by your **kmeans** algorithm. Because **kmeans** is sensitive to initialization, repeat your **kmeans** at least 5 times with different random initializations and show the plot of each initialization.
# To quantitatively evaluate the clustering performance, we evaluate the $\textit{unsupervised clustering accuracy}$, which can be written as follows,
# $$
# \text{accuracy} = \max_{\mathcal M} \frac{\sum_{i=1}^{n} \mathbb{I}(y_i = \mathcal M(z_i))}{n}, n = 1000,
# $$
# where $y_i$ is the ground-truth label, $z_i$ is the cluster assignment produced by the algorithm,
# and $\mathcal M$ ranges over all possible one-to-one mapping between clusters and labels and $\mathbb{I}(x)$
# is a indicator function ($\mathbb{I}(x) = 1 \text{if}~x=1; \text{otherwise} ~0$).
# Please use the __accuracy_score__ function defined below to calculate the accuracy.
# Report the best clustering accuracy you get out of 10 random initializations. 

def accuracy_score(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    assert y_true.shape == y_pred.shape, 'illegal inputs'
 
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) * 1.0 / y_pred.size
  
 def kmeans2(X,  Initial, k, max_iter = 500):
    
    assert len(X) > k, 'illegal inputs'

    centers = Initial    
    new_centers = np.zeros((k, X.shape[1]))

    labels = np.zeros(len(X))

    from scipy.spatial import distance
    for i in range(max_iter):
        H = distance.cdist(X, centers, 'euclidean')

        for j in range(len(X)):
            labels[j] = np.argsort(H[j])[0]
    
        for j in range(k):
            new_centers[j] = X[labels==j].mean(axis=0)
            
        if np.sum(new_centers == centers) == k:
            break
        centers = new_centers
        
    return labels
  
  
  import pandas as pd
tmp = ["random.seed","Accuracy"]
n = 11
k = 4
df = pd.DataFrame(np.zeros([len(tmp),n]), index = tmp)
Initial0 = np.array([[0, 2],[2, 0],[4, 4],[8, 0]])

for i in range(0,n):
    
    if i < n-1:
        np.random.seed(i)
        idx = np.random.choice(len(X), k, replace=False)
        Initial = X2d[idx]
    else:
        Initial = Initial0 
    
    y_pred = kmeans2(X2d,Initial,k).astype(int)
    
    df.loc[tmp[0],i] = i
    df.loc[tmp[1],i] = accuracy_score(y, y_pred)

df.loc[tmp[0],:] = df.loc[tmp[0],:].apply("{:.0f}".format)
df.loc[tmp[0],n-1] = "NA"
MA = max(df.loc[tmp[1],:])

print("k="+str(k))
print(tmp[0]+" is variable from 0 to "+str(n-2))
print(tmp[0]+"=NA: Manually enter the following initial values")
print(Initial0)

print("\n the best clustering accuracy: ="+str(MA))

plt.plot(df.loc[tmp[0],:], df.loc[tmp[1],:])
plt.xlabel(tmp[0])
plt.ylabel(tmp[1])
plt.yticks([0.74,0.745,0.75])
plt.show()

df

i_max = df.loc["random.seed",df.loc["Accuracy",:]==MA][0]
np.random.seed(int(i_max))
idx = np.random.choice(len(X), k, replace=False)
Initial = X2d[idx]

Xs = X2d
ys = kmeans2(X2d,Initial,k)

for i, _ in enumerate(ys):
    if ys[i] == 0:
        plt.plot([Xs[i, 0]], [Xs[i, 1]], 'r.')
    elif ys[i] == 1:
        plt.plot([Xs[i, 0]], [Xs[i, 1]], 'b.')
    elif ys[i] == 2:
        plt.plot([Xs[i, 0]], [Xs[i, 1]], 'g.')
    elif ys[i] == 3:
        plt.plot([Xs[i, 0]], [Xs[i, 1]], 'y.')

for i in range(0,len(Initial)):
    plt.plot([Initial[i, 0]], [Initial[i, 1]], 'k.', markersize=12)

print("Visualize the result: random.seed="+str(i_max)+" (the best clustering accuracy)")
print("*Black=Initial point")
plt.show()
