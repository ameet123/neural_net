from keras.models import Sequential
from keras.layers import Dense
from keras import layers
import numpy as np
import os
from keras import activations
from keras import losses
from keras import optimizers
from keras import metrics

np.random.seed(7)

filename = "data/pima-indians-diabetes.data.csv"
data = np.loadtxt(filename, delimiter=',')
# Create X and Y arrays
X = data[:, 0:8]
Y = data[:, 8]
boundary = int(round(X.shape[0] * .8))
X_train = X[:boundary, :]
X_test = X[boundary:, :]
Y_train = Y[:boundary]
Y_test = Y[boundary:]

print("X Values=>{}".format(X[0]))
print("Y Values=>{}".format(Y[0]))

# Activation vs loss
# activation is used to compute feed forward propagation time node values or activations.
# loss is the cost function, J(theta), which we try to minimize.

# Model creation
model = Sequential()
model.add(Dense(12, input_dim=8, activation=activations.relu))
model.add(Dense(8, activation=activations.relu))
model.add(Dense(1, activation=activations.sigmoid))
# Compile

model.compile(optimizer='adam', loss=losses.binary_crossentropy, metrics=['accuracy'])

# Fit
model.fit(X_train, Y_train, batch_size=10, epochs=150, verbose=0)
# Evaluation
scores = model.evaluate(X_test, Y_test)
print("loss={} {}={}  ".format(scores[0], model.metrics_names[1], scores[1]))
