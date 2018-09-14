import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

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
iterations = 100
batch_size = 2000
LABEL_POS = 14
INPUT_NEURONS = 128
HIDDEN_NEURONS = 64
HIDDEN_LAYERS = 10
DEFAULT_DROPOUT = 0.2
scalarX = RobustScaler()
train_file = "data\eye_state.csv"
REST_ACT = 'relu'
OUTPUT_ACT = 'sigmoid'

X_train, X_test, Y_train, Y_test = readNp(train_file, scalarX, 14)
class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
INPUT_DIM = X_train.shape[1]


def define_model(init='glorot_normal', dropout=DEFAULT_DROPOUT, hidden_layers=HIDDEN_LAYERS):
    model = Sequential()
    model.add(Dense(INPUT_NEURONS, input_dim=INPUT_DIM, kernel_initializer=init, activation=REST_ACT))
    model.add(Dropout(dropout))
    for i in range(hidden_layers):
        model.add(Dense(HIDDEN_NEURONS, kernel_initializer=init, activation=REST_ACT))
        model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init, activation=OUTPUT_ACT))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=define_model, verbose=0, batch_size=batch_size, epochs=iterations,
                        class_weight=class_weights)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X_train, Y, cv=kfold)

# define list of parameters
init = ['glorot_uniform', 'glorot_normal']
dropout = [0.2, 0.3]
hidden_layers_list = [8, 10]
param_grid = dict(hidden_layers=hidden_layers_list, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, Y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
fit_times = grid_result.cv_results_['mean_fit_time']
score_times = grid_result.cv_results_['mean_score_time']
for mean, stdev, fit_time, score_time, param in zip(means, stds, fit_times, score_times, params):
    print("%f (%f) fit:%f sec. score:%f sec. with: %r" % (mean, stdev, fit_time, score_time, param))

print("Classification report")
y_pred = np.rint(grid.predict(X_test))
print(classification_report(Y_test, y_pred))
print("\nConfusion Matrix:\n")
matrix = confusion_matrix(Y_test, y_pred)
print(matrix)
