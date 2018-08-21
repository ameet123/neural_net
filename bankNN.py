'''
adding more neurons helps.
higher epoch of 250 helps to improve as well.
smaller batchsize helps
b=25 => accuracy on 1: 21
b=1000 => accuracy on 1:12

'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from pandas_ml import ConfusionMatrix
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data\\bank\\bank.csv", sep=';')

# Convert to numerical
factorize_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'y']
for c in factorize_cols:
    if c == 'y':
        offset = 0
    else:
        offset = 1
    df[c] = df[c].factorize()[0] + offset

# remove unwanted features
unwanted = ['day', 'month', 'duration']
df = df.drop(unwanted, axis=1)

# Scale the features
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
xDF = df.iloc[:, 0:-1]
yDF = df.iloc[:, -1].to_frame()
x_normDF = pd.DataFrame(scalarX.fit_transform(xDF), columns=xDF.columns)
y_normDF = pd.DataFrame(scalarY.fit_transform(yDF), columns=yDF.columns)

print("Records in set:{} # of features:{}".format(x_normDF.shape[0], x_normDF.shape[1]))

X_train, X_test, Y_train, Y_test = train_test_split(x_normDF.values, y_normDF.values, test_size=0.2, random_state=42)
Y_train = np.rint(Y_train)
Y_test = np.rint(Y_test)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1], kernel_initializer='normal'))
model.add(Dense(128, activation='relu', kernel_initializer='normal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit
model.fit(X_train, Y_train, batch_size=25, epochs=250, verbose=0)
# Evaluation
scores = model.evaluate(X_test, Y_test)
print("loss={} {}={}  ".format(scores[0], model.metrics_names[1], scores[1]))

# Get confusion matrix
Y_pred = np.rint(model.predict(X_test))

matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)

print("Shape of Y-test:{} pred:{}".format(Y_test.shape, Y_pred.shape))
cm = ConfusionMatrix(Y_test.reshape(Y_test.size), Y_pred.reshape(Y_pred.size))
print(cm)
cm.plot()
plt.show()
