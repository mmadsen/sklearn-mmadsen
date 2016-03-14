import seaborn as sns

def confusion_heatmap(y_test, y_pred, class_labels = None, axis_labels = None, title = None, transparent = False, reverse_color = False, filename = None):
    """
    Draw a Seaborn heatmap from a sklearn confusion matrix, with optional
    text labels.  Returns the matplotlib AX object for use later, or saving
    to a file.

    Optional axis labels and title parameters are available; axis labels have reasonable defaults if not given, and
    no title is shown by default.

    For use in slide presentations, there is an option for making the graphic background transparent when saving it
    to a file, and to reverse the color scheme of axis elements to white for placing the graphic on a dark background.

    :param y_test:
    :param y_pred:
    :param class_labels: optional
    :param axis_labels: optional
    :param title: optional
    :param transparent: optional
    :param reverse_color: optional
    :param filename: optional
    :return:
    """
    from sklearn.metrics import confusion_matrix

    if reverse_color is True:
        custom_style = {'axes.labelcolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white'}
        sns.set_style("darkgrid", rc=custom_style)


    mat = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                     xticklabels=class_labels, yticklabels=class_labels)

    if axis_labels is None:
        xlabel = "Predicted Class"
        ylabel = "Actual Class"
    else:
        xlabel = axis_labels[0]
        ylabel = axis_labels[1]

    if title is not None:
        ax.set_title(title)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if filename is not None:
        fig = ax.get_figure()
        fig.savefig(filename, dpi=300, transparent=transparent)

    return ax