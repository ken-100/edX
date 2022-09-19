# Problem 1
## Dataset Generation
Write a function to **generate a training set** of size $m$
- randomly generate a weight vector $w \in \mathbb{R}^{10}$, normalize length
- generate a training set $\{(x_i , y_i)\}$ of size m
  - $x_i$: random vector in $\mathbb{R}^{10}$ from $\textbf{N}(0, I)$
  - $y_i$: $\{0, +1\}$ with $P[y = +1] = \sigma(w \cdot x_i)$ and $P[y = 0] = 1 - \sigma(w \cdot x_i)$
  
## Algorithm 1: logistic regression
The goal is to learn $w$.  Algorithm 1 is logistic
  regression (you may use the built-in method LogisticRegression for this. Use max_iter=1000).
  
  
## Algorithm 2: gradient descent with square loss
Define square loss as
$$L_i(w^{(t)}) = \frac{1}{2} \left( \sigma(w^{(t)} \cdot x) - y_i \right)^2$$

  Algorithm 2 is
  gradient descent with respect to square loss (code this
  up yourself -- run for 1000 iterations, use step size eta = 0.01).
  
  
## Algorithm 3: stochastic gradient descent with square loss
Similar to gradient descent, except we use the gradient at a single random training point every iteration.

The gradient of the loss is given by $$ \nabla_w L(w) = (\sigma(w \cdot x) - y) \sigma'(w \cdot x) x. $$ The derivative of the sigmoid function satisfies $$ \sigma'(t) = \frac{e^{-t}}{(1 + e^{-t})^2} = \frac{1}{1 + e^{-t}} - \frac{1}{(1 + e^{-t})^2} = \sigma(t) - \sigma(t)^2. $$ Denoting $\sigma(w \cdot x)$ by $\hat{y}$, this means that one convenient way to simplify the gradient above is as follows: $$ \nabla_w L(w) = (\hat{y} - y)(1 - \hat{y})\hat{y} x. $$


## Evaluation
Measure error $\|w - \hat{w}\|_2$ for each method at different sample size. For any
  fixed value of $m$, choose many different $w$'s and average the
  values $\|w - 
  \hat{w}\|_2$ for Algorithms 1, 2 and 3.  Plot the results
  for for each algorithm as you make $m$ large (use $m=50, 100, 150, 200, 250$).
  Also record, for each algorithm, the time taken to run the overall experiment.
  

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd


def generate_data(m):
    # generate weight vector
    w_star = np.random.normal(size=(10,))
    w_star /= np.linalg.norm(w_star)
    
    # training sample
    X = np.zeros((m, 10))
    Y = np.zeros((m,))
    for i in range(m):
        X[i, :] = np.random.normal(size=(10,))
        prob = 1.0 / (1.0 + np.exp(-w_star.dot(X[i])))
        Y[i] = np.random.choice([0.0, 1.0], p=[1-prob, prob])
    return w_star, X, Y

# scikit-learn Cheat-sheet 
# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

def log_reg(X, Y):
    return LogisticRegression(max_iter=1000).fit(X, Y).coef_
# https://qiita.com/0NE_shoT_/items/c42d8093e2fed9bf1b7a

def gradient(w, xi, yi):
    yhat = 1.0 / (1.0 + np.exp(-w.dot(xi)))
    return (yhat - yi) * yhat * (1 - yhat) * xi


def gd(X, Y, n_iter=1000, eta=0.01):
    w = np.zeros(X.shape[1])
    for t in range(n_iter):
        grad = np.mean([gradient(w, x, y) for x, y in zip(X, Y)], axis=0)
        w = w - eta * grad
    return w
# https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/%E7%B5%B1%E8%A8%88%E5%AD%A6/%E5%8B%BE%E9%85%8D%E9%99%8D%E4%B8%8B%E6%B3%95%E3%81%A7%E9%87%8D%E5%9B%9E%E5%B8%B0%E5%88%86%E6%9E%90%E3%81%97%E3%81%A6%E3%81%BF%E3%81%9F/#toc3
# https://helve-blog.com/posts/math/gradient-descent-armijo/

def sgd(X, Y, n_iter=1000, eta=0.01):
    w = np.zeros(X.shape[1])
    for t in range(n_iter):
        i = np.random.randint(X.shape[0])
        w = w - eta * gradient(w, X[i], Y[i])
    return w
# https://qiita.com/koshian2/items/894a8ca6f2d5ec9ab70a
# https://note.com/kanawoinvestment/n/n8c439a9a2736

ms = [50, 100, 150, 200, 250]
def run_trials(model, num_trials=10):
    errs = []
    for m in ms:
        trial_errs = []
        for _ in range(num_trials):
            w, X, Y = generate_data(m)
            w_hat = model(X, Y)
            trial_errs.append(np.linalg.norm(w - w_hat))
        errs.append(np.mean(trial_errs))
    return errs
    
#%%time
logreg_errs = run_trials(log_reg)

#%%time
gd_errs = run_trials(gd)

#%%time
sgd_errs = run_trials(sgd)


A = ["Alg1:Logistic","Alg2:GD","Alg3:SGD"]
plt.plot(ms, logreg_errs, label='Logistic regression')
plt.plot(ms, gd_errs, label='GD')
plt.plot(ms, sgd_errs, label='SGD')

plt.xlabel('sample size $m$')
plt.ylabel('error ($\|w - \hat{w}\|$)')
plt.legend()
plt.show()

E =  pd.DataFrame(np.zeros([len(A),len(ms)]),index=A,columns=ms)
E.iloc[0,:] = logreg_errs
E.iloc[1,:] = gd_errs
E.iloc[2,:] = sgd_errs
E
