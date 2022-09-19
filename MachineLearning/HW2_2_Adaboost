For each depth in $1, \dots, 5$, instantiate an AdaBoost classifier with the base learner set to be a decision tree of that depth, and then record the 10-fold cross-validated error on the entire breast cancer data set. Plot the resulting curve of accuracy against base classifier depth. Use $101$ as your random state for both the base learner as well as the AdaBoost classifier every time.


from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

cancer = datasets.load_breast_cancer()

accuracy = []
for depth in range(1, 6):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=101)
    ada = AdaBoostClassifier(base_estimator=dt, n_estimators=10, learning_rate=1, random_state=101)
    accuracy.append(np.mean(cross_val_score(ada, cancer.data, cancer.target, cv=10)))

plt.plot(range(1, 6), accuracy)
plt.xlabel('base dt depth')
plt.ylabel('accuracy')
