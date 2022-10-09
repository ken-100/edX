import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Problem 1
# In this problem we will look at image compression using SVD, following the lines of the well-known "Eigenfaces" experiment. The basic concept is to represent an image (in grayscale) of size $m \times n$ as an $m \times n$ real matrix $M$. SVD is then applied to this matrix to obtain $U$, $S$, and $V$ such that $M = U S V^T$. Here $U$ and $V$ are the matrices whose columns are the left and right singular vectors respectively, and $S$ is a diagonal $m \times n$ matrix consisting of the singular values of $M$. The number of non-zero singular values is the rank of $M$. By using just the largest $k$ singular values (and corresponding left and right singular vectors), one obtains the best rank-$k$ approximation to $M$.

data = datasets.fetch_olivetti_faces()
images = data.images

# Returns the best rank-k approximation to M
def svd_reconstruct(M, k):
    # TODO: Complete this!
    # Advice: pass in full_matrices=False to svd to avoid dimensionality issues
    u,s,vh = svd(M)
    uk = u[:,:k]
    vhk = vh[:k,:]
    sk = np.diag(s[:k])
    krank_image = uk.dot(sk).dot(vhk)
    return krank_image

def errorl1(org_image,app_image):
    error = np.abs(org_image-app_image).mean()
    return error

k_errors = []
k_term = range(1,31)
for k in k_term:
    kimages = [svd_reconstruct(org_img,k) for org_img in images]
    tmp = [errorl1(org_img,app_img) for org_img,app_img in zip(images,kimages)]
    tmp = sum(tmp)/len(tmp)
    k_errors.append(tmp)
    
plt.plot(k_term,k_errors)
plt.xlabel("k")
plt.ylabel("Error")
plt.title("Average Reconstruction Error")
plt.show()

# (b) Pick any image in the dataset, and display the following side-by-side as images: the original, and the best rank-$k$ approximations for $k = 10, 20, 30, 40$. You will find the `imshow` method in matplotlib useful for this; pass in `cmap='gray'` to render in grayscale. Feel free to play around further.

j=20
fig,ax = plt.subplots(4,2,figsize=(8,16))
image = fetch_olivetti_faces()["images"][j]

for i,k in enumerate([10,20,30,40]):
    k_image = svd_reconstruct(images[j],k)
    
    ax[i,0].imshow(image,cmap='gray')
    ax[i,0].set_title("Orignal")
    ax[i,0].axis("off")
    
    ax[i,1].imshow(k_image,cmap='gray')
    ax[i,1].set_title("Appromation(k={})".format(k))
    ax[i,1].axis("off")
    
plt.show()


# In this problem we visualize the Wisconsin breast cancer dataset in two dimensions using PCA. First, rescale the data so that every feature has mean 0 and standard deviation 1 across the various points in the dataset. You may find `sklearn.preprocessing.StandardScaler` useful for this. Next, compute the top two principal components of the dataset using PCA, and for every data point, compute its coordinates (i.e. projections) along these two principal components. You should do this in two ways:
# 1. By using SVD directly. Do not use any PCA built-ins.
# 2. By using `sklearn.decomposition.PCA`.

# The two approaches should give exactly the same result, and this also acts as a check. (But note that the signs of the singular vectors may be flipped in the two approaches since singular vectors are only determined uniquely up to sign. If this happens, flip signs to make everything identical again.)

# Your final goal is to make a scatterplot of the dataset in 2 dimensions, where the x-axis is the first principal component and the y-axis is the second. Color the points by their diagnosis (malignant or benign). Do this for both approaches. Your plots should be identical. Does the data look roughly separable already in 2 dimensions?

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data,columns=load_breast_cancer().feature_names)

X = df.values
y = list(cancer.target)
df["label"] = y

X_std = StandardScaler().fit_transform(X) #Standardization
print("StandardizationX std :",np.std(X_std))
print("StandardizationX mean :",round(np.mean(X_std),2))
df.head()


def pcaSVD(X):
  u,sigma,vt = np.linalg.svd(X, full_matrices=False) # SVD
  return (X @ vt.T), (sigma**2) / (len(X)-1)

eig_pairs =  pcaSVD(X_std)
Y_SVD = np.hstack((-eig_pairs[0][:,0].reshape(len(X_std), 1), eig_pairs[0][:,1].reshape(len(X_std), 1)))
    
df_SVD = pd.DataFrame(Y_SVD[:,0:2],columns=["PC1", "PC2"])
df_SVD["Label"] = y
df_SVD["Label"] = df_SVD["Label"].replace(0, "Benign").replace(1, "Malignant")

A = df_SVD[df_SVD["Label"]=="Benign"]
B = df_SVD[df_SVD["Label"]=="Malignant"]

plt.scatter(A["PC1"], A["PC2"], label="Benign")
plt.scatter(B["PC1"], B["PC2"], label="Malignant")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("1. By using SVD directly")
plt.legend()

df_SVD.head()



sklearn_pca = PCA(n_components=3)
Y_sklearn = sklearn_pca.fit_transform(X_std)

df_sklearn = pd.DataFrame(Y_sklearn[:,0:2],columns=["PC1", "PC2"])
df_sklearn["Label"] = y
df_sklearn["Label"] = df_sklearn["Label"].replace(0, "Benign").replace(1, "Malignant")


scatter = plt.scatter(df_sklearn["PC1"], df_sklearn["PC2"], c=y, alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(handles=scatter.legend_elements()[0], labels=list(cancer.target_names))

df_sklearn.head()
