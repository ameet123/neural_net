import keras
import pandas as pd
from numpy.random import seed
from sklearn.preprocessing import RobustScaler
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
seed(42)
OUTPUT_POS = 0
OUTPUT_COL = 'class'
train_file = "data\\training.csv"
test_file = "data\\testing.csv"


# Functions
def readX_Y(file, scalar_x, label_col):
    dataDF = pd.read_csv(file, sep=',')
    uniq_classes = dataDF[label_col].unique().size
    print("Records:{} features:{}".format(dataDF.shape[0], dataDF.shape[1] - 1))
    # Get X and Y separately,
    x_DF = dataDF.iloc[:, 1:]
    y_DF = dataDF.iloc[:, 0].to_frame()
    if (scalar_x is not None):
        x_DF = pd.DataFrame(scalar_x.fit_transform(x_DF), columns=x_DF.columns)
    y_DF[label_col] = y_DF[label_col].factorize()[0]
    # y_train_arr = keras.utils.to_categorical(y_DF.values, y_DF[label_col].unique().size).astype(int)
    x_train_arr = x_DF.values
    return x_train_arr, y_DF.values, uniq_classes


# Execution
scalarX = RobustScaler()
X_train, Y_train, uniq_classes = readX_Y(train_file, None, OUTPUT_COL)


model = svm.SVC(kernel='linear')
svm_fit = model.fit(X_train,Y_train)
print(svm_fit)
