#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

X, y = load_iris(return_X_y=True)

plt.scatter(X[:, 1], X[:, 2], c=y)
plt.show()


# In[2]:


import numpy as np

data = open("leukemia_data.txt", 'r').read()

data = data.split("\n")
headers = data[0].split()
data = [[float(y) for y in x.split()] for x in data[1:]]
data = np.array(data)

y_1 = data[:, 0]
X_1 = data[:, 1:]

print(headers)
print(y_1)
print(X_1)


# In[3]:


def get_column(i, data):
    return list(map(lambda x: x[i], data))

num_variables = len(headers) - 1
fig, axs = plt.subplots(3, 2, squeeze=False)
plt.subplots_adjust(wspace=0.4,hspace=0.8)

y = get_column(0, data)
for i in range(num_variables):
    x = get_column(i + 1, data)
    ax = axs[i % 3, 0 if i < 3 else 1]
    ax.scatter(x, y)
    ax.set_xlabel(headers[i + 1])
    ax.set_ylabel(headers[0])
    plt.scatter(x, y)
plt.show()


# In[19]:


from logistic_regression import LogisticRegression

model = LogisticRegression()

result = model.fit(X_1, y_1)
print(result)

y_hat = model.predict(X_1)
print("accuracy my implementation:", sum(y_hat.round() == y_1)/len(y_1))


# In[18]:


from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

sklearn_model = SklearnLogisticRegression(random_state=0).fit(X_1, y_1)
sklearn_y_hat = sklearn_model.predict(X_1)
print("accuracy sklearn implementation:", sum(sklearn_y_hat.round() == y_1)/len(y_1))


# In[22]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from knn_classifier import KnnClassifier

data = open("breast-cancer-wisconsin.data.txt").read()

df = pd.read_csv("breast-cancer-wisconsin.data.txt", header=None)
col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion', 
             'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

df.columns = col_names
df.drop('Id', axis=1, inplace=True)
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')

# Just dropping nan gives accuracy of 0.9708
df.dropna(inplace=True)

X = df.drop(['Class'], axis=1)
Y = df['Class']

# Interpolate instead of dropping nan gives accuracy 0.9714
# for col in X.columns:
#         col_median=X[col].median()
#         X[col].fillna(col_median, inplace=True)

X = np.array(X)
Y = np.array(Y)

# #This is bad but it works for now
transformed_data = np.zeros(X.shape)
for col_i in range(X.shape[1]):
    current_column = X[:, col_i]
    mu = np.mean(current_column)
    sigma = np.std(current_column)    
    normalized_column = np.array((current_column - mu)/sigma)
    transformed_data[:, col_i] = normalized_column


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# TODO: Support making predictions on multiple rows like sklearn model
p = 2
knn_classifier = KnnClassifier(K=3, p=p)
num_correct = 0
Y_pred_my_model = []
for x, y in zip(X_test, Y_test):
    result = knn_classifier.predict(X_train, Y_train, x)
    Y_pred_my_model.append(result)
Y_pred_my_model = np.array(Y_pred_my_model)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

Y_pred_sklearn_model = knn.predict(X_test)

from sklearn.metrics import accuracy_score
print('My model accuracy score: {0:0.4f}'. format(accuracy_score(Y_test, Y_pred_my_model)))
print('Sklearn model accuracy score: {0:0.4f}'. format(accuracy_score(Y_test, Y_pred_sklearn_model)))


# In[ ]:




