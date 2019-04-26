import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import model_selection
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA

from time import time

def all_same(items):
  return len(set(items)) == 1

# Load training data from csv file
data = pd.read_csv("data/train.csv")

# Extract feature columns
feature_cols = list(data.columns[1:])

# Extract target column 'label'
target_col = data.columns[0]

# Separate the data into feature data and target data (X and y, respectively)
X = data[feature_cols]
y = data[target_col]

# Apply PCA by fitting the data with only 60 dimensions
pca = PCA(n_components=60).fit(X)
# Transform the data using the PCA fit above
X = pca.transform(X)
y = y.values


# Shuffle and split the dataset into the number of training and testing points above
sss = model_selection.StratifiedShuffleSplit( n_splits=3, test_size=0.4, random_state=42)
sss.get_n_splits(X, y)
for train_index, test_index in sss.split(X, y):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

# Fit a KNN classifier on the training set
knn_clf = KNeighborsClassifier(n_neighbors=3, p=2)
knn_clf.fit(X_train, y_train)

# Initialize the array of predicted labels
y_pred = np.empty(len(y_test), dtype=np.int)

start = time()

# Find the nearest neighbors indices for each sample in the test set
kneighbors = knn_clf.kneighbors(X_test, return_distance=False)

# For each set of neighbors indices
for idx, indices in enumerate(kneighbors):
  # Find the actual training samples & their labels
  neighbors = [X_train[i] for i in indices]
  neighbors_labels = [y_train[i] for i in indices]
  
  # if all labels are the same, use it as the prediction
  if all_same(neighbors_labels):
    y_pred[idx] = neighbors_labels[0]
  else:
    # else fit a SVM classifier using the neighbors, and label the test samples
    svm_clf = svm.SVC(C=0.5, kernel='rbf', decision_function_shape='ovo', random_state=42, gamma='auto')
    svm_clf.fit(neighbors, neighbors_labels)
    label = svm_clf.predict(X_test[idx].reshape(1, -1))

    y_pred[idx] = label
    
 
end = time()

#Result metrics

from sklearn.metrics import classification_report, confusion_matrix 
print("###########################################") 
print(confusion_matrix(y_test,y_pred))  
print("###########################################") 
print(classification_report(y_test,y_pred)) 
print("###########################################") 
print(accuracy_score(y_test, y_pred))
print("###########################################") 
print("Made predictions in {:.4f} seconds.".format(end - start))
print("###########################################") 
      
#Predicted Data Visualization
#from sklearn.utils.multiclass import unique_labels
def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_test, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names=["0","1","2","3","4","5","6","7","8","9"]
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()

