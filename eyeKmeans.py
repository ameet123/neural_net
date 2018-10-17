import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import parallel_coordinates
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn import metrics

# Eye state
eye_train_file = "data\\eye_state.csv"
scalar_x = RobustScaler()
eye_label_pos = 14
eye_label_name='eyeDetection'
eyeDF = pd.read_csv(eye_train_file, sep=',')
#Scale for plot
scaleDF = eyeDF.iloc[:,:-1]
scaleDF = pd.DataFrame(scalar_x.fit_transform(scaleDF), columns=scaleDF.columns)
scaleDF[eye_label_name] = eyeDF[eye_label_name]

# 1. parallel plot
parallel_coordinates(scaleDF, eye_label_name)
plt.legend(loc='lower left', frameon=False)
plt.savefig('image/km/eye/eye_det_parallel.png')
plt.clf()
plt.cla()

eyeX = eyeDF.drop(eye_label_name,axis=1)
eyeX = pd.DataFrame(scalar_x.fit_transform(eyeX), columns=eyeX.columns).values
eyeY = eyeDF.iloc[:, 0].values
print("Rows,Cols in the eye Detection dataset:{}".format(eyeX.shape))
eyekm = KMeans(n_clusters=2)
eyekm.fit(eyeX)
y_eyekm = eyekm.predict(eyeX)

# Create DF
eyekmDF = pd.DataFrame()
eyekmDF['data_index'] = eyeDF.index.values
eyekmDF['cluster'] = eyekm.labels_

# Visualize
for i in range(2, eyeX.shape[1]):
    plt.scatter(eyeX[:, 1], eyeX[:, i], c=y_eyekm, s=50, cmap='viridis')
    plt.savefig('image/km/eye/eye_km_scatter_1-' + str(i) + '.png')
