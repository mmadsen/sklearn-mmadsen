import networkx as nx
import numpy as np
from sklearn.utils import shuffle

def graphs_to_eigenvalue_matrix(graph_list, num_eigenvalues = None):
    """
    Given a list of NetworkX graphs, returns a numeric matrix where rows represent graphs,
    and columns represent the reverse sorted eigenvalues of the Laplacian matrix for each graph,
    possibly trimmed to only use the num_eigenvalues largest values.  If num_eigenvalues is
    unspecified, the maximum number of eigenvalues are used such that all of the graphs in the
    list have at least that number of eigenvalues.  This compensates for situations where some of the
    graphs in the list have smaller numbers of vertices and thus eigenvalues than the rest.
    """
    # in a combined data set, the graphs may be of very different sizes, so we should
    # first get all the eigenvalues and then figure out how many to return
    eigen_rows = []
    min_size = np.inf
    for ix in range(0, len(graph_list)):
        spectrum = sorted(nx.spectrum.laplacian_spectrum(graph_list[ix], weight=None), reverse=True)
        num_nonzero = np.count_nonzero(spectrum)
        min_size = min(min_size, num_nonzero)
        eigen_rows.append(spectrum)






    # we either use all of the eigenvalues, or we use the smaller of
    # the requested number or the actual number (if it is smaller than requested)
    if num_eigenvalues is None:
        ev_used = min_size
    else:
        ev_used = min(min_size, num_eigenvalues)

    print "(debug) eigenvalues - minimum shared eigenvalues: %s num_eigenvalues: %s ev_used: %s" % (min_size, num_eigenvalues, ev_used)

    data_mat = np.zeros((len(graph_list),ev_used))
    #print "data matrix shape: ", data_mat.shape

    for ix in range(0, len(graph_list)):
        data_mat[ix,:] = eigen_rows[ix][0:ev_used]

    return data_mat


def graph_train_test_split(graph_list, label_list, test_fraction=0.20):
    """
    Randomly splits a set of graphs and labels into training and testing data sets.  We need a custom function
    because the dataset isn't a numeric matrix, but a list of NetworkX Graph objects.  In case there is class
    structure (i.e., we filled the arrays first with instances of one class, then another class...) we consistently
    shuffle both lists.
    """

    graph_list, label_list = shuffle(graph_list, label_list)

    rand_ix = np.random.random_integers(0, len(graph_list)-1, size=int(len(graph_list) * test_fraction))
    print "random indices: %s" % rand_ix
    print "min ix: %s  max ix: %s" % (min(rand_ix), max(rand_ix))

    test_graphs = []
    test_labels = []

    train_graphs = []
    train_labels = []

    # first copy the chosen test values, without deleting anything since that would alter the indices
    for ix in rand_ix:
        test_graphs.append(graph_list[ix])
        test_labels.append(label_list[ix])

    # now copy the indices that are NOT in the test index list
    for ix in range(0, len(graph_list)):
        if ix in rand_ix:
            continue
        train_graphs.append(graph_list[ix])
        train_labels.append(label_list[ix])

    return (train_graphs, train_labels, test_graphs, test_labels)