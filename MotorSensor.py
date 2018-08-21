from timeit import default_timer as timer
from matplotlib import pyplot as plt
import keras
import numpy as np
import pandas as pd
from keras import losses
from keras.layers import Dense
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


def readX_Y(file, scalar_x, label_pos):
    '''
    read file into DataFrame and extract X and Y numpy arrays,
    normalize them, one-hot-encode the labels and return
    :param file:
    :param scalar_x:
    :param label_col:
    :return:
    '''
    dataDF = pd.read_csv(file, sep=' ', header=None)
    # Get X and Y separately,
    x_DF = dataDF.drop(dataDF.columns[[label_pos]], axis=1)
    y_DF = dataDF.iloc[:, label_pos].to_frame()
    uniq_classes = y_DF[y_DF.columns[0]].unique().size
    # normalizing
    x_DF = pd.DataFrame(scalar_x.fit_transform(x_DF), columns=x_DF.columns)

    # y_DF[label_col] = y_DF[label_col].factorize()[0]
    y_DF[y_DF.columns[0]] = y_DF[y_DF.columns[0]].factorize()[0]
    # one hot encoding of labels.
    y_train_arr = keras.utils.to_categorical(y_DF.values, uniq_classes)
    x_train_arr = x_DF.values
    print("Records:{} features:{} classes:{}".format(dataDF.shape[0], dataDF.shape[1] - 1, uniq_classes))
    return x_train_arr, y_train_arr, uniq_classes


iterations = 150
batch_size = 50
scalarX = MinMaxScaler()
train_file = "data\Sensorless_drive_diagnosis.txt"
X, Y, num_classes = readX_Y(train_file, scalarX, -1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

features = X_train.shape[1]
# Model building
model = Sequential()
model.add(Dense(128, input_dim=features, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Compile
start = timer()
model.compile(optimizer='adam', loss=losses.categorical_crossentropy, metrics=['accuracy'])
print("Model compilation done:{} sec.".format(round(timer() - start, 2)))
start = timer()
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=iterations, verbose=0)
print("Model FIT done:{} sec.".format(round(timer() - start, 2)))
# Print history metrics
plt.plot(history.history['acc'])
plt.show()

# Evaluation
start = timer()
score = model.evaluate(X_test, Y_test, verbose=0)
print("Model evaluation done:{} sec.".format(round(timer() - start, 2)))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Get confusion matrix
Y_pred = np.rint(model.predict(X_test))

print("Shape of Y-test:{} pred:{}".format(Y_test.shape, Y_pred.shape))
auc = roc_auc_score(Y_test, Y_pred)
prec_score = precision_score(Y_test, Y_pred, average='micro')
print("AUC score:{} Precision score:{}".format(auc, prec_score))
