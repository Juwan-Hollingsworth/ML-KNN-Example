import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

#training & test samples
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

# a = [1,1,1,2,2,3,4,5,6]
# from collections import Counter
# most_common = Counter(a).most_common(1)
# print(most_common[0][0])

from knn import KNN
clf = KNN(k=3) #create a classifier 
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)
