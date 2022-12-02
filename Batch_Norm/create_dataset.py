from sklearn.datasets import make_circles 
from numpy import where 
from matplotlib import pyplot as plt 


#making cirlces 
X,y = make_circles(n_samples = 1200, noise = 0.15, random_state = 1)

#select indices of the points which belongs to each class. 
for i in range(2):
    idx = where(y==i)
    plt.scatter(X[idx, 0], X[idx, 1], label = str(i))

plt.legend()
plt.savefig("./binary_class_data.png")
plt.show()


