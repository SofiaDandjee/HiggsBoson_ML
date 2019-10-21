import numpy as np
import matplotlib.pyplot as plt


def plot_cross_validation(extra_param, extra_param_name, best_extra_param, rmse_train, rmse_test, loss):
    """Visualization of the curves of rmse_train and rmse_test depending on the extra parameter (lambda or gamma)."""
    plt.plot(extra_param, rmse_train, color='b', label="Train error")
    plt.plot(extra_param, rmse_test, color='g', label="Test error")
    plt.plot(best_extra_param, loss, 'r*', markersize=15, label="Minimal loss")
    plt.xlabel(extra_param_name)
    plt.ylabel("RMSE")
    plt.title("Cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("Cross validation")
