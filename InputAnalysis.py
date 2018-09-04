import numpy as np
from matplotlib import pyplot as plt

file = "data\eye_state.csv"
dataset = np.loadtxt(file, delimiter=',', skiprows=1)
label_position = 14

X = dataset[:, 0:label_position]
y_full = dataset[:, label_position]

print("X shape:{}".format(X.shape))


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

plot_variables()