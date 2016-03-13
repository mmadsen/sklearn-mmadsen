import seaborn as sns

def confusion_heatmap(y_test, y_pred, labels):
    from sklearn.metrics import confusion_matrix
    mat = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                     xticklabels=labels, yticklabels=labels)