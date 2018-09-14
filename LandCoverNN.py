from timeit import default_timer as timer

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from numpy.random import seed
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight

seed(42)
from tensorflow import set_random_seed

set_random_seed(42)


# Functions
def readX_Y(file, scalar_x, label_col):
    dataDF = pd.read_csv(file, sep=',')
    uniq_classes = dataDF[label_col].unique().size
    print("Records:{} features:{}".format(dataDF.shape[0], dataDF.shape[1] - 1))
    # Get X and Y separately,
    x_DF = dataDF.iloc[:, 1:]
    y_DF = dataDF.iloc[:, 0].to_frame()
    # x_DF = pd.DataFrame(scalar_x.fit_transform(x_DF), columns=x_DF.columns)
    y_DF[label_col] = y_DF[label_col].factorize()[0]
    # one hot encoding of labels.
    y_train_arr = keras.utils.to_categorical(y_DF.values, y_DF[label_col].unique().size).astype(int)
    x_train_arr = x_DF.values
    c_wt = class_weight.compute_class_weight('balanced', [0, 1, 2, 3, 4, 5], [y.argmax() for y in y_train_arr])
    print("Class wth:{}".format(c_wt))
    return x_train_arr, y_train_arr, uniq_classes, c_wt


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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Visualize loss history
def plot_loss_history(plot_file, history):
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    epoch_count = range(1, len(training_loss) + 1)
    plt.title("Train vs. Test Loss/Accuracy")
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.plot(history.history['acc'], 'c.')
    plt.plot(history.history['val_acc'], 'm.')
    plt.legend(['Training Loss', 'Test Loss', 'train Acc', 'test Acc'], loc='center right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.grid()
    plt.savefig(plot_file)
    plt.clf()


# Global Variables
iterations = 100
batch_size = 1000
scalarX = RobustScaler()
OUTPUT_POS = 0
OUTPUT_COL = 'class'
train_file = "data\\training.csv"
OUTPUT_ACTIVATION = 'softmax'
INUPUT_NEURONS = 256
HIDDEN_NEURONS = 128
HIDDEN_LAYERS = 10

# Execution
X_train, Y_train, uniq_classes, train_class_weights = readX_Y(train_file, scalarX, OUTPUT_COL)
print("Number of uniq classes:{}".format(uniq_classes))
model = define_model(X_train.shape[1], INUPUT_NEURONS, uniq_classes, HIDDEN_NEURONS, HIDDEN_LAYERS, True,
                     OUTPUT_ACTIVATION)

# Test
test_file = "data\\testing.csv"
scalar_test_X = RobustScaler()
X_test, Y_test, uniq_test_classes, test_classes = readX_Y(test_file, scalar_test_X, OUTPUT_COL)

start = timer()
# class_weight=train_class_weights
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=iterations,
                    validation_data=(X_test, Y_test), verbose=0)
print("NN Model fit in: {} sec.".format(round(timer() - start, 2)))

plot_loss_history('LandCover_test_train_loss_acc.png', history)
#
# Evaluation
start = timer()
score = model.evaluate(X_test, Y_test, verbose=0)
print("Model evaluation done:{} sec. Test loss:{} accuracy:{}".format(round(timer() - start, 2), score[0], score[1]))
Y_pred = np.rint(model.predict(X_test))
