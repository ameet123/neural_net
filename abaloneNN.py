import numpy as np
import pandas as pd
from keras import activations
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
np.random.seed(10)
cols = ['Sex', 'len', 'diameter', 'ht', 'whole weight', 'shucked wt', 'viscera wt', 'shell wt', 'rings']
df = pd.read_csv("data\\abalone.data", names=cols)
# Pre processing
# convert sex to categorical and normalize
df['Sex'] = df['Sex'].factorize()[0]



min_max = MinMaxScaler()
normDF = pd.DataFrame(min_max.fit_transform(df), columns=df.columns)

print("Records in set:{}".format(normDF.shape[0]))

X = normDF.values[:, 0:8]
Y = normDF.values[:, 8]
# division into train test
boundary = int(round(X.shape[0] * .8))
X_train = X[:boundary, :]
X_test = X[boundary:, :]
Y_train = Y[:boundary]
Y_test = Y[boundary:]

features = X.shape[1]

print("Size of train:{} test:{} # of features:{}".format(X_train.shape[0], X_test.shape[0], features))

# Build model
model = Sequential()
model.add(Dense(128, input_dim=features, activation=activations.relu))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Fit
model.fit(X_train, Y_train, batch_size=32, epochs=50, verbose=0)

# Evaluation
scores = model.evaluate(X_test, Y_test)
print("loss={} {}={}  ".format(scores[0], model.metrics_names[1], scores[1]))

# Prediction
obs = 20
pred_ind = np.arange(obs)
X_pred = X_test[pred_ind].reshape((pred_ind.size, features))
Y_pred = model.predict(X_pred).reshape((pred_ind.size, 1))
d_pred = np.hstack((X_pred, Y_pred))
Y_inv = min_max.inverse_transform(d_pred)[:, features]
predDF = df.rings[0:obs].to_frame()
predDF = predDF.rename(columns={'rings': 'actual'})

predDF['pred'] = Y_inv.tolist()
predDF.pred = predDF.pred.astype(int)
print("Shape of pred:{} inveser:{}".format(d_pred.shape, Y_inv.shape))
print(predDF)
