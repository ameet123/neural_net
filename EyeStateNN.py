from timeit import default_timer as timer

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler

import seaborn as sn
import matplotlib.pyplot as plt

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


from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)
iterations = 200
batch_size = 1000
scalarX = RobustScaler()
train_file = "data\eye_state.csv"
X_train, X_test, Y_train, Y_test = readNp(train_file, scalarX, 14)

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
    model.add(Dense(out_neurons, kernel_initializer='normal', activation=output_act))
    return model

model = define_model(X_train.shape[1], 128, 1, 64, 10, True, 'sigmoid')

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Class weights
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=iterations, class_weight=class_weights, verbose=0)

# Evaluation
start = timer()
score = model.evaluate(X_test, Y_test, verbose=0)
print("Model evaluation done:{} sec.".format(round(timer() - start, 2)))
print('Test loss:{} accuracy:{}', score[0], score[1])

print(model.predict(X_test))
Y_pred = np.rint(model.predict(X_test))
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
# Visualize matrix

cm_df = pd.DataFrame(matrix,index=[ i for i in "01"],columns=[i for i in "01"])
sn.set(font_scale=1.4)
sn_plot = sn.heatmap(cm_df,annot=True,annot_kws={"size":16}, fmt='g')
# plt.savefig('robust_scaling_NN_heatmap.png')
plt.show()