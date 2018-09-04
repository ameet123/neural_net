import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler

file = "data\eye_state.csv"
dataset = np.loadtxt(file, delimiter=',', skiprows=1)
label_position = 14

X = dataset[:, 0:label_position]
y_full = dataset[:, label_position]

print("X shape:{}".format(X.shape))

# Standard Scaler
ssX = StandardScaler()
ssX = ssX.fit(X)
X1 = ssX.transform(X)
# Robust
rsX = RobustScaler()
rsX = rsX.fit(X)
X2 = rsX.transform(X)


def var_plot(X, v, fig, subplt):
    ind = np.arange(X.shape[0])
    ax = fig.add_subplot(subplt)
    pl1 = ax.scatter(ind, X[:, v] / 1000, marker='.', s=6, color=np.random.rand(3, 1))
    ax.set_title('Var-' + str(v), fontsize=12)
    ax.tick_params(axis='both', labelsize=8)
    l = ax.set_xticklabels(ind, rotation=45)


def plot_variables():
    fig = plt.figure(1)
    for i in range(9):
        var_plot(X, i, fig, int('33' + str(i + 1)))
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def set_x_format(ax):
    ax.tick_params(axis='both', labelsize=8)
    l = ax.set_xticklabels(ind, rotation=45)


# plot_variables()
ind = np.arange(X.shape[0])
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Var-0 vs. Var-1\n", fontsize=12)

ax[0].scatter(X[:, 0], X[:, 1], color=np.where(y_full > 0, 'r', 'b'))
ax[1].scatter(X1[:, 0], X1[:, 1], color=np.where(y_full > 0, 'r', 'b'))
ax[2].scatter(X2[:, 0], X2[:, 1], color=np.where(y_full > 0, 'r', 'b'))
for a in ax:
    set_x_format(a)
ax[0].set_title("Unscaled data", fontsize=12)
ax[1].set_title("After standard scaling (zoomed in)", fontsize=12)
ax[2].set_title("After robust scaling (zoomed in)", fontsize=12)

# for the scaled data, we zoom in to the data center (outlier can't be seen!)
for a in ax[1:]:
    a.set_xlim(-3, 3)
    a.set_ylim(-3, 3)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("image\eyeState_v0_v1.png")
plt.show()
