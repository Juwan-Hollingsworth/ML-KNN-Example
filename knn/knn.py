import numpy as np
from collections import Counter
# Concept of KNN:
# A sample is classified by a popularity vote from it's neaerest neighbors
# Distance is calculated by Euclidean distance 

# define Euclidean distance method 
# Takes 2 vectors 
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


# define class
class KNN:
    # k = number of nearest neighbors to consider
    def __init__(self, k=3):
        self.k = k  

    # fit method = to fit the training samples - training step
    def fit(self,x,y):
        self.X_train = x
        self.y_train = y
    
#    predict method = predict new samples
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute all distances using Euclidean distance 
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k-nearest samples, labels
        # np.argsort = sort the distances & return the indices 
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote (most common class)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

        #


