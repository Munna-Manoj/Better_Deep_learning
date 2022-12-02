"""
training and testing the MLP over the binary classification problem circle dataset generated using scikit-learn.
At first, We check it without Batch Normalization.
"""

from sklearn.datasets import make_circles 
import tensorflow as tf 
from matplotlib import pyplot as plt 


#Creating the synthetic Dataset.
X,Y = make_circles(n_samples = 1500, noise = 0.15, random_state = 2)

#creating the train/test split
n_train = 1000 
trainX, testX = X[:n_train, :], X[n_train:, :] 
trainY, testY = Y[:n_train], Y[n_train:]

#Creating our DNN model architecture.

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu , kernel_initializer = 'he_uniform', input_shape = (2,)))
model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)) 
opt = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9)
model.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])


#fit the model with train and test dataset.
history = model.fit(trainX, trainY, validation_data = (testX, testY), epochs = 100, verbose = 0)

#evaluation of the model.
_, train_acc = model.evaluate(trainX, trainY, verbose = 0)
_, test_acc = model.evaluate(testX, testY, verbose = 0)
print("Trai Acc: %.3f, Test_Accc: %.3f"%(train_acc, test_acc)) 


#plotting loss curve during learning.
plt.subplot(211)
plt.title('Cross-Entropy Loss', pad = -45)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label = 'test')
plt.legend()


#plotting accuracy curve during learning.
plt.subplot(212)
plt.title('Accuracy', pad = -45)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label = 'test')
plt.legend()
plt.show()


