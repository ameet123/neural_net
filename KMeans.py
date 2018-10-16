import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.tools.plotting import parallel_coordinates
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

sns.set()

train_file = "data\\training.csv"
label_pos = 0
scalar_x = RobustScaler()
dataDF = pd.read_csv(train_file, sep=',')

# Data info
dDF = dataDF.copy()
# change col names to shorter words
feature_cols = dDF.columns[1:]
simple_feat_cols = ['c' + str(p) for p, q in enumerate(feature_cols)]
simple_feat_cols.insert(0, 'class')
dDF.columns = simple_feat_cols
# make 2 parts of df
dDF1 = dDF.iloc[:, 0:14]
dDF2 = dDF.iloc[:, 15:]
dDF2['class'] = dDF['class']
# 1. parallel plot
parallel_coordinates(dDF1, 'class')
plt.legend(loc='lower left', frameon=False)
plt.savefig('image/km/parallel_plot1.png')
plt.clf()
plt.cla()

parallel_coordinates(dDF2, 'class')
plt.legend(loc='lower left')
plt.savefig('image/km/parallel_plot2.png', frameon=False)

X = dataDF.iloc[:, 1:]
Y = dataDF.iloc[:, 0].values
# Scaling
X = pd.DataFrame(scalar_x.fit_transform(X), columns=X.columns).values

print("Rows in the LandCover dataset:{} type of X:{}".format(X.shape[0], type(X)))
km = KMeans(n_clusters=6)
km.fit(X)
y_km = km.predict(X)

# Create DF
kmDF = pd.DataFrame()
kmDF['data_index'] = dataDF.index.values
kmDF['cluster'] = km.labels_
kmDF['real'] = dataDF.iloc[0]

# Visualize
# for i in range(1, X.shape[1]):
#     plt.scatter(X[:, 0], X[:, i], c=y_km, s=50, cmap='viridis')
#     plt.savefig('km_scatter_0-' + str(i) + '.png')
