
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
import matplotlib.pyplot as plt2
import matplotlib.cm as cm

# Load training data from csv file
try:
    data = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print("Digits dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")
    
    
# Calculate number of samples
n_images = data.shape[0]

# Calculate number of features
n_features = data.shape[1] - 1

# Plot number of occurrences of each label
plt2 = data.label.groupby(data.label).count().plot(kind="bar")
plt2.set_xlabel("digit label")
plt2.set_ylabel("number of occurrences")

# Print the results
print("Total number of samples: {}".format(n_images))
print("Number of features: {}".format(n_features))


images = test.iloc[:,:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))

import matplotlib.pyplot as plt
# display image

def display(img):
    
    # (784) => (28,28)
    one_image = img.reshape(28, 28)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
   

# output image

display(images[5])
X=images[5].reshape(28,28)

# compute the average intensity (the average value of a pixel in an image)
data.intensity = X.mean(axis=0)
data.intensity=pd.DataFrame(data.intensity)

# Plot average intensity of each label
plot = data.intensity.groupby(data.label).mean().plot(kind="bar")
plot.set_xlabel("digit label")
plot.set_ylabel("average intensity")

print(data.intensity.describe())