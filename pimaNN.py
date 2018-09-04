'''
Outlines a progression of trying to achieve better performance on binary classification.
1. normalize the data -> improves accuracy
2. round the predictions as well as the train/test labels to integers
3. start with a single hidden layer.
4. # of neurons in the input layer, somewhere in the vicinity of # of features or a few multiples of that
5. final layer activation func -> sigmoid
6. optimizer 'adam'
'''
import numpy as np
from keras import activations
from keras.layers import Dense
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

filename = "data/pima-indians-diabetes.data.csv"
data = np.loadtxt(filename, delimiter=',')
# Create X and Y arrays
X = data[:, 0:8]
Y = data[:, 8]
# Feature Scaling
scalarX = MinMaxScaler()
scalarX = scalarX.fit(X)
X = scalarX.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

print("X Values=>{}".format(X[0]))
print("Y Values=>{}".format(Y[0]))

# Activation vs loss
# activation is used to compute feed forward propagation time node values or activations.
# loss is the cost function, J(theta), which we try to minimize.

# Model creation
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='normal', activation=activations.relu))
model.add(Dense(8, kernel_initializer='normal', activation=activations.relu))
model.add(Dense(1, kernel_initializer='normal', activation=activations.sigmoid))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit
model.fit(X_train, Y_train, batch_size=10, epochs=100, verbose=0)
# Evaluation
scores = model.evaluate(X_test, Y_test)
print("loss={} {}={}".format(scores[0], model.metrics_names[1], scores[1]))

Y_pred = np.rint(model.predict(X_test))
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
