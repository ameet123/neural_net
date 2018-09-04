from timeit import default_timer as timer

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler


def readBinary(file, scalar_x, label_pos):
    dataDF = pd.read_csv(file, sep=',')
    x_DF = dataDF.drop(dataDF.columns[[label_pos]], axis=1)
    y_DF = dataDF.iloc[:, label_pos].to_frame()
    x_DF = pd.DataFrame(scalar_x.fit_transform(x_DF), columns=x_DF.columns)
    print(x_DF.head())
    print(y_DF.head())
    return x_DF.values.astype('float32'), y_DF.values


def readNp(file, scalarX, label_pos):
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    X = data[:, 0:label_pos]
    Y = data[:, label_pos]
    scalarX = scalarX.fit(X)
    X = scalarX.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)
    return X_train, X_test, Y_train, Y_test


iterations = 200
batch_size = 1000
scalarX = RobustScaler()
train_file = "data\eye_state.csv"
X_train, X_test, Y_train, Y_test = readNp(train_file, scalarX, 14)
print("X rows:{} features:{} Y rows:{}".format(X_train.shape[0], X_train.shape[1], Y_train.shape[0]))


def define_model(input_dim, in_neurons, out_neurons, hidden_dim, num_hidden_layer, is_dropout, output_act,
                 other_act='relu'):
    model = Sequential()
    model.add(Dense(in_neurons, input_dim=input_dim, kernel_initializer='normal', activation=other_act))
    if is_dropout:
        model.add(Dropout(0.2))
    for i in range(num_hidden_layer):
        model.add(Dense(hidden_dim, kernel_initializer='normal', activation=other_act))
        if is_dropout:
            model.add(Dropout(0.2))
    print("Adding output layer.")
    model.add(Dense(out_neurons, kernel_initializer='normal', activation=output_act))
    return model


print("Shape of X:{}".format(X_train.shape))
model = define_model(X_train.shape[1], 256, 1, 128, 20, True, 'sigmoid')

model.summary()

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Class weights
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=iterations,
                    class_weight=class_weights, verbose=0)

# Evaluation
start = timer()
score = model.evaluate(X_test, Y_test, verbose=0)
print("Model evaluation done:{} sec.".format(round(timer() - start, 2)))
print('Test loss:{} accuracy:{}', score[0], score[1])

print(model.predict(X_test))
Y_pred = model.predict(X_test).astype(int)
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
