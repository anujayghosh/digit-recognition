# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:40:56 2019

@author: Anujay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('data/train.csv')
y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:].values


'''m sklearn.model_selection import train_test_split
# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)

'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#train_img, test_img, train_label, test_label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_img)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#import numpy as np
#X_train = np.array(X_train).reshape(-1,1)
sc.fit(X_train)
X_train= sc.transform(X_train)
#X_test= np.array(X_test).reshape(-1,1)
X_test = sc.transform(X_test)


from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(0.95)

pca.fit(X_train)


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=314)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance_r = pca.explained_variance_ratio_
explained_variance= pca.explained_variance_

ccc=pca.n_components_
hhh=[0]*ccc
iii=[0]*ccc
total=0
for ii in range(ccc):
    total+=explained_variance_r[ii]*100
    hhh[ii]=total
    iii[ii]=ii
    
import matplotlib.pyplot as plt
plt.plot(iii,hhh)
plt.show()

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()