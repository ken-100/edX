c = ['x1', 'x2']
X = np.array([
    [1, 1],
    [2, 1],
    [6, 1],
    [1, 3],
    [2, 3],
    [6, 3]])
X = pd.DataFrame(X, columns=c)
print(X)
#    x1  x2
# 0   1   1
# 1   2   1
# 2   6   1
# 3   1   3
# 4   2   3
# 5   6   3

def f(i):
    # model = KMeans(n_clusters=2, random_state=0, max_iter=1, init='random')  # initを省略すると、k-means++法が適応される(randomではk-means法が適応)
    model = KMeans(n_clusters=2, max_iter=i, init=X0)
    model.fit(X)
    clusters = model.predict(X)

    X1 = pd.DataFrame(model.cluster_centers_ ,columns = c)
    
    return X1

def g(X0):
    I = 2
    J = 2
    fig, ax = plt.subplots(I, J, squeeze=False,figsize=(8,4),tight_layout=True)

    k=0
    for i in range(I):
        for j in range(J):

            if k == 0:
                Y = X0
            else:
                Y = f(k)
                exec(f"X{k}=Y.copy()")

            ax[i,j].scatter(X["x1"], X["x2"])
            ax[i,j].scatter(Y["x1"], Y["x2"])
            ax[i,j].set_xticks(list(range(7)))
            ax[i,j].set_yticks(list(range(5)))
            ax[i,j].set_title("X"+str(k))
            k += 1

    plt.show()
    
    
X0 = np.array([
       [0, 2],
       [3, 2]])
X0 = pd.DataFrame(X0, columns=c)
g(X0)



X0 = np.array([
       [1.5, 2],
       [1.5, 3]])
X0 = pd.DataFrame(X0, columns=c)
g(X0)
