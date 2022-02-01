# Q2.1_graded
# Do not change the above line.

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

class RBF:
    def __init__(self, input_dims, hidden_layer_neurons):
        self.centers = np.zeros((hidden_layer_neurons, input_dims))
        self.W = np.random.random((hidden_layer_neurons))
        self.hidden_layer_neurons = hidden_layer_neurons
        self.spreads = np.zeros((1, hidden_layer_neurons))
        self.model = Sequential()

    def gaussian_function(self, x, c, s):
        return np.exp(-1 * ((np.linalg.norm(x - c) / s) ** 2))

    def calc_activation(self, X):
        g_values = np.zeros((self.hidden_layer_neurons, X.shape[1]))
        for i in range(X.shape[1]):
            x = X[:,i]
            g = np.zeros((self.hidden_layer_neurons, ))
            for c in range(self.hidden_layer_neurons):
                center = self.centers[c]
                s = self.spreads[c]
                g[c] = self.gaussian_function(x, center, s)
            g_values[:,i] = g
        return g_values

    def fit(self, X, Y):
        self.centers = KMeans(n_clusters=self.hidden_layer_neurons).fit(X.T).cluster_centers_
        avg_distance = 0
        for c in range(len(self.centers)):
            distance_to_nearest = max([np.linalg.norm(self.centers[c] - X[:,i]) for i in range(X.shape[1])])
            avg_distance += distance_to_nearest
        avg_distance /= self.hidden_layer_neurons
        self.spreads = np.repeat(avg_distance, self.hidden_layer_neurons)
        g_values = self.calc_activation(X)
        self.model.add(layers.Dense(3, input_dim=self.hidden_layer_neurons, activation='softmax'))
        sgd_optimizer = SGD(
            learning_rate=0.3,
        )
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=sgd_optimizer,
                metrics=['accuracy'],
        )
        history = self.model.fit(
            x=g_values.T,
            y=to_categorical(Y),
            epochs=100,
            shuffle=True,
        )
    def predict(self, X):
        g_values = self.calc_activation(X)
        p = self.model.predict(g_values.T)
        return np.argmax(p, axis=-1)

# Q2.1_graded
# Do not change the above line.
X = np.array([[.3, .4, .6, .7, .2, .3, .4, .5, .6, .7, .8, .2, .3, .4, .5, .6, .7, .8, .2, .3, .4, .5, .6, .7, .8, .4, .5, .6], 
              [.3, .3, .3, .3, .4, .4, .4, .4, .4, .4, .4, .5, .5, .5, .5, .5, .5, .5, .6, .6, .6, .6, .6, .6, .6, .7, .7, .7]])
Y = np.array([[2, 2, 2, 2, 2, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 1, 2, 1, 1, 1]]).reshape(-1)

# Q2.1_graded
# Do not change the above line.
r = RBF(2, 3)
r.fit(X, Y)

# Q2.1_graded
# Do not change the above line.
test_x1 = np.random.uniform(low=0.1, high=0.9, size=(1, 30))
test_x2 = np.random.uniform(low=0.1, high=0.8, size=(1, 30))
test_data = np.zeros((2, 30))
test_data[0,:] = test_x1
test_data[1,:] = test_x2
predicted = r.predict(test_data)

# Q2.1_graded
# Do not change the above line.
plt.scatter(np.concatenate((X[0, :], test_data[0, :])), np.concatenate((X[1, :], test_data[1, :])), c=np.concatenate((Y, predicted)), s=40, cmap=plt.cm.Spectral)

