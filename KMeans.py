from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.tools.plotting import parallel_coordinates
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

sns.set()

# Land Cover
N_CLUSTERS = 6
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
km = KMeans(n_clusters=N_CLUSTERS)
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

land_score = metrics.adjusted_rand_score(Y, km.labels_)
land_mutual_info_score = metrics.adjusted_mutual_info_score(Y, km.labels_)
print(
    "Land Cover kmeans  AdjustedRand Index:{} Adjusted Mutual Info score:{}".format(land_score, land_mutual_info_score))

# Cluster name mapping
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'cyan', 'red']

fig = plt.figure()
fig.suptitle("LandCover KMeans", fontsize=12)
x = 0
y = 0
nrows = 2
ncols = 3
isLegend = True
for cluster in range(N_CLUSTERS):
    assignment = Y[np.where(y_km == cluster)]
    uniq_vals, count = np.unique(assignment, return_counts=True)
    val_cnt = np.column_stack((uniq_vals, count))
    data = Counter(assignment)
    common_val = data.most_common(1)[0][0]
    # plot
    ax1 = plt.subplot2grid((2, 3), (x, y))
    explode = np.array([0.2 if x == np.argmax(count) else 0 for x in range(count.size)])
    pie = plt.pie(count, explode=explode, shadow=True, startangle=140, colors=colors)
    if isLegend:
        fig.legend(pie[0], uniq_vals, loc="lower left",  fontsize=9, frameon=False)
        isLegend = False
    plt.title("[{}] -> {}".format(cluster, common_val), fontsize=10)
    y = y + 1
    if (y == ncols):
        x = x + 1
        y = 0
    print('cluster: {} -> most common={}\t total classes={}\t members={}'.format(cluster, common_val, uniq_vals.size,
                                                                                 assignment.size))
    print("\t\t{}\n".format(np.array2string(val_cnt)))

png_file = "image/km/LandCover/km_clust_pie.png"
plt.savefig(png_file)
