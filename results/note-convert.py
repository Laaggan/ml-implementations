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

