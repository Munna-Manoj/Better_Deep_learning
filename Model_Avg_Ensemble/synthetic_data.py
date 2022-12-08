from sklearn.datasets import make_blobs 
from matplotlib import pyplot 
from numpy import where 


#generating the 2D synthetic classification dataset. 
X,y = make_blobs(n_samples = 500, centers = 3, n_features = 2, cluster_std = 2, random_state = 2) 

#plot the scatter plot 
for class_value in range(3):
    row_idx = where(y == class_value)
    pyplot.scatter(X[row_idx,0], X[row_idx, 1]) 
pyplot.show()