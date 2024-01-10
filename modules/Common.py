import matplotlib.pylab as plt


def plot_grid(histories, labels, acc_name, loss_name):
    fig = plt.figure(figsize=(20, 10))
    # Validation Loss Graph
    axis = fig.add_subplot(1, 2, 1)
    for history, label in zip(histories, labels):
        axis.set_title("Validation Loss")
        axis.plot(history.history[loss_name], label="Val Loss - {}".format(label))
    axis.legend()
    axis.grid()
    # Validation Accurary Graph
    axis = fig.add_subplot(1, 2, 2)
    for history, label in zip(histories, labels):
        axis.set_title("Validation Accuracy")
        axis.plot(history.history[acc_name], label="Val Acc - {}".format(label))
    axis.legend()
    axis.grid()
    plt.show()
