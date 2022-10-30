import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree, ensemble
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Assignment Constants
RS = 10 #Random_State
FIGSIZE = (5,3)

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns = cancer.feature_names)

target = pd.DataFrame(cancer.target, columns=['class'])
ratio = 1 - sum(target["class"]) / len(target)

print("shape = " ,df.shape)
print( "the base rate of malignant cancer occurrence :" , "{:.3f}".format(ratio) )


##### Decision Tree
clf = tree.DecisionTreeClassifier(max_depth=2,random_state=RS)
clf = clf.fit(cancer.data, cancer.target)
clf.score(cancer.data, cancer.target)

plt.figure(dpi=150)
plot_tree(clf, feature_names=df.columns, class_names=True)

print("Example: Decision Tree, max_depth=2")
plt.show()



##### Cross_Validation
tmp = 10
tmp1 = "Max depth"
R = pd.DataFrame(np.zeros([1,tmp]),index=[tmp1]) #Result

for i in range(0,tmp):

    clf = tree.DecisionTreeClassifier(max_depth=i+1,random_state=RS)
    clf.fit(cancer.data, cancer.target)   #<- for just Score1

    R.loc[tmp1,i] = i+1
    R.loc["Score1",i] = clf.score(cancer.data, cancer.target) #Full Set
    R.loc["Score2",i] = np.mean(cross_val_score(clf, cancer.data, cancer.target, cv=KFold(n_splits=10,random_state=RS,shuffle=True))) #Cross-Validated

R.iloc[0,:] = R.iloc[0,:].apply("{:.0f}".format)

x = R.loc[tmp1 ,:]
plt.figure(figsize=FIGSIZE)
plt.xlabel(tmp1)
plt.ylabel("Accuracy")
plt.plot(x, R.loc["Score1",:], label="All_Data")
plt.plot(x, R.loc["Score2",:], label="Cross_Validation")
plt.legend(fontsize="small")

for i in [1,2]:
    R.iloc[i,:] = R.iloc[i,:].apply("{:.3f}".format)

print("The accuracy")
R



##### Importance factor
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=10)
clf = DecisionTreeClassifier(max_depth=2, random_state=RS)
clf.fit(X_train, y_train)
 
# print(clf.feature_importances_)
df2 = pd.DataFrame(
    {'feature':cancer.feature_names, 'importance':clf.feature_importances_})
 
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.3)
ax.barh(df2.feature, df2.importance)
print("Example: Importance factor, max_depth=2")
plt.show()


##### Hold Out
tmp = 10

tmp1 = "Max depth"
R = pd.DataFrame(np.zeros([1,tmp]),index=[tmp1])

for i in range(0,tmp):

    clf = tree.DecisionTreeClassifier(max_depth=i+1,random_state=RS)
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.25, stratify=cancer.target, random_state=RS)
    clf.fit(X_train, y_train)
    
    R.loc[tmp1,i] = i+1
    R.loc["Training score",i] = clf.score(X_train, y_train)
    R.loc["Test score",i] = clf.score(X_test, y_test)

    clf = clf.fit(df, target)
    predicted = clf.predict(df)   
    R.loc["Match",i] = sum(predicted == target["class"]) / len(target)

R.iloc[0,:] = R.iloc[0,:].apply("{:.0f}".format)

x = R.loc[tmp1,:]
plt.figure(figsize=FIGSIZE)
plt.xlabel(tmp1)
plt.ylabel("Test score")
plt.plot(x, R.loc["Training score",:],label="Training")
plt.plot(x, R.loc["Test score",:],label="Test")
plt.legend(fontsize="large")

tmp = R.loc["Max depth",R.loc["Test score",:] == max(R.loc["Test score",:])]

for i in range(1,len(R)):
    R.iloc[i,:] = R.iloc[i,:].apply("{:.3f}".format)

print("for your reference: Hold out test")

print("Best Max depth = " ,int(tmp), "(Based on Test score)")
R


##### Consider n_estimators
tmp = 20
tmp1 = ["Max_depth","n_estimators","Average score"] + ["Test" + str(i) for i in range(1,11)]
tmp2 = "n_estimators"
MD = 5 #max_depth

R = pd.DataFrame(np.zeros([len(tmp1),tmp]),index=tmp1) #Result
R.loc["Max_depth",:] = MD

for i in range(0,tmp):

    kf = KFold(n_splits=10, shuffle=True, random_state=RS)
    forest = RandomForestClassifier(max_depth = MD, n_estimators=i+1, random_state=RS)
    dtc_scores = cross_val_score(forest, cancer.data, cancer.target, cv=kf)

    R.loc[tmp2,i] = i+1
    R.loc[tmp1[3:],i] = dtc_scores
    R.loc["Average score",i] = np.mean(dtc_scores)

x = R.loc[tmp2,:]
y = R.loc["Average score",:]
plt.figure(figsize=FIGSIZE)
plt.xlabel(tmp2)
plt.ylabel("Average score")
plt.xticks( np.arange(0, 20, 4))
plt.yticks( np.arange(0.92, 0.971, 0.01))
plt.plot(x, y)
plt.show()

tmp = R.loc[tmp2,R.loc["Average score",:] == max(R.loc["Average score",:])]
tmp = list(tmp)[0]

for i in range(0,2):
    R.iloc[i,:] = R.iloc[i,:].apply("{:.0f}".format)
for i in range(2,len(R)):
    R.iloc[i,:] = R.iloc[i,:].apply("{:.3f}".format)

print("10-fold cross-validated accuracy.")
print("Max_depth =", MD, " (fixed)")
print("Best ",tmp2,"=" ,int(tmp))
R


##### Consider Max_depth
tmp = 10
tmp1 = ["Max_depth","n_estimators","Average score"] + ["Test" + str(i) for i in range(1,11)]
tmp2 = "Max_depth"
NE = 16 #n_estimators

R = pd.DataFrame(np.zeros([len(tmp1),tmp]),index=tmp1) #Result
R.loc["n_estimators",:] = NE

for i in range(0,tmp):


    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    forest = RandomForestClassifier(max_depth = i+1, n_estimators=NE, random_state=10)
    dtc_scores = cross_val_score(forest, cancer.data, cancer.target, cv=kf)

    R.loc[tmp2,i] = i+1
    R.loc[tmp1[3:],i] = dtc_scores
    R.loc["Average score",i] = np.mean(dtc_scores)

x = R.loc[tmp2,:]
y = R.loc["Average score",:]
plt.figure(figsize=FIGSIZE)
plt.xlabel(tmp2)
plt.ylabel("Average score")
plt.xticks( np.arange(0, 10, 2))
plt.yticks( np.arange(0.92, 0.971, 0.01))
plt.plot(x, y)
plt.show()

tmp = R.loc[tmp2,R.loc["Average score",:] == max(R.loc["Average score",:])]
tmp = list(tmp)[0]

for i in range(0,2):
    R.iloc[i,:] = R.iloc[i,:].apply("{:.0f}".format)
for i in range(2,len(R)):
    R.iloc[i,:] = R.iloc[i,:].apply("{:.3f}".format)

print("10-fold cross-validated accuracy.")
print("n_estimators =" , NE , "(fixed)")
print("Best ",tmp2,"=" ,int(tmp))
R
