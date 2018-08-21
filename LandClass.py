import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense,Dropout
from keras.models import Sequential
from pandas_ml import ConfusionMatrix
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import losses

def readX_Y(file,scalar_x, label_col):
    '''
    read file into DataFrame and extract X and Y numpy arrays,
    normalize them, one-hot-encode the labels and return
    :param file:
    :param scalar_x:
    :param label_col:
    :return:
    '''
    dataDF = pd.read_csv(file, sep=',')
    uniq_classes=dataDF[label_col].unique().size
    print("Records:{} features:{}".format(dataDF.shape[0], dataDF.shape[1] - 1))
    # Get X and Y separately,
    x_DF = dataDF.iloc[:, 1:]
    y_DF = dataDF.iloc[:, 0].to_frame()
    x_DF= pd.DataFrame(scalar_x.fit_transform(x_DF), columns=x_DF.columns)
    y_DF[label_col] = y_DF[label_col].factorize()[0]
    # one hot encoding of labels.
    y_train_arr = keras.utils.to_categorical(y_DF.values, y_DF[label_col].unique().size)
    x_train_arr = x_DF.values
    return x_train_arr,y_train_arr,uniq_classes

iterations=150
batch_size = 50
scalarX  = MinMaxScaler()
train_file="data\\training.csv"
X_train,Y_train,num_classes=readX_Y(train_file,scalarX,'class')

features = X_train.shape[1]
# Model building
model = Sequential()
model.add(Dense(128,input_dim=features,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation='softmax'))
model.summary()

# Compile
model.compile(optimizer='adam',loss=losses.categorical_crossentropy,metrics=['accuracy'])
history  = model.fit(X_train,Y_train,batch_size=batch_size,epochs=iterations,verbose=0)

# Evaluation
# Get test set
test_file="data\\testing.csv"
scalar_test_X = MinMaxScaler()
X_test,Y_test,test_classes=readX_Y(test_file,scalar_test_X,'class')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


