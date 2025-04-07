import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y, grid_size=(100, 100), n_classes=3, ax=None, offset=0.5):
    x_min, x_max = X[:, 0].min() - offset, X[:, 0].max() + offset
    y_min, y_max = X[:, 1].min() - offset, X[:, 1].max() + offset

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size[0]),
        np.linspace(y_min, y_max, grid_size[1])
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    colors = ListedColormap(["#FFAAAA", "#AAAAFF", "#AAFFAA"][:n_classes])

    if ax is None:
        ax = plt.gca()  

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=colors)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=colors)
    ax.set_xlabel("Característica 1")
    ax.set_ylabel("Característica 2")

    model_name = type(model).__name__
    if model_name == "SVC":
        model_name += f" ({model.kernel})"  

    ax.set_title(f"Modelo: {model_name}")
    