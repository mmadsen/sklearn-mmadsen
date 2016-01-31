#!/usr/bin/env python
# Copyright (c) 2015.  Mark E. Madsen <mark@madsenlab.org>
#
# This work is licensed under the terms of the Apache Software License, Version 2.0.  See the file LICENSE for details.

"""
Description here

"""


import numpy as np
import pprint as pp
import networkx as nx
from sklearn.base import BaseEstimator

from .mathutils import weighted_mode

def check_list_of_graphs(l):
    if isinstance(l, list) == False:
        raise ValueError
    if isinstance(l[0], nx.Graph) == False:
        raise ValueError


class GraphEigenvalueNearestNeighbors(BaseEstimator):
    """Classifier implementing the k-nearest neighbors vote for NetworkX graphs, with distance calculated as Laplacian spectral distance.

    Parameters
    ----------

    n_neighbors : int, optional (default = 10)
        Number of neighbors to use by default for :meth:`predict` queries.

    spectral_fraction: float, optional (default = 0.9)
        Fraction of the total sum of Laplacian eigenvalues to use in calculating the spectral distance.  By default,
        we use the k largest eigenvalues where their sum is greater than 90%.  Passing None uses the entire
        Laplacian spectrum.


    """

    def __init__(self, spectral_fraction = 0.9, n_neighbors = 10):
        self.num_neighbors = n_neighbors
        self.spectral_fraction = spectral_fraction
        pass



    def fit(self, X, y):
        """Store the training graphs and their associated class labels.

        Knn algorithms don't actually have a "fitting" step since they're really just lookup tables.
        So we just store the training data in lookup tables for later prediction.  The method exists solely to
        conform to the scikit-learn API to allow :class:GraphEigenvalueNearestNeighbors to be used
        in :class:Pipeline or in cross-validation.

        Parameters
        ----------
        X : array-like, shape (n_samples)
            Training samples.

        y : array-like, shape (n_samples)
            Training class labels.

        Returns
        -------
        self

        """

        check_list_of_graphs()

        self.id_to_graph_map = dict()
        self.id_to_class_map = dict()
        self.num_training_samples = len(X)
        for i in range(0, len(X)):
            self.id_to_graph_map[i] = X[i]
            self.id_to_class_map[i] = y[i]

        return self


    def predict(self, X):
        """Predict the class labels for the provided graphs.

        Parameters
        ----------
        X : array-like, shape (n_query)
            Test samples.

        Returns
        -------
        y : array of shape [n_samples]
            Class labels for each data sample.
        """

        check_list_of_graphs()

        # vector for the predicted classes
        y_pred = []

        # vector as long as the number of neighbors we consider, that we will fill with the smallest
        # distances we find, bumping out larger values as we go.
        # and we use uniform weights
        weights = np.ones(self.num_neighbors)
        smallest_dist = np.full(self.num_neighbors, np.inf)
        graph_for_neighbor_map = dict()
        graph_ids = self.id_to_graph_map.keys()

        for target_graph in X:
            # make a pass through all the training graphs, calculating distance to the target

            for id in graph_ids:
                train_graph = self.id_to_graph_map[id]
                train_graph_dist = self._graph_spectral_similarity(target_graph, train_graph, self.spectral_fraction)

                # see if the distance is one of the smallest on the list
                indices = np.argmax(smallest_dist)
                # initially it'll be several until we fill up the list the first time
                if indices.size == 1:
                    if train_graph_dist < smallest_dist[indices]:
                        smallest_dist[indices] = train_graph_dist
                        graph_for_neighbor_map[indices] = id
                else:
                    if train_graph_dist < smallest_dist[indices[0]]:
                        smallest_dist[indices[0]] = train_graph_dist
                        graph_for_neighbor_map[indices[0]] = id

                print smallest_dist

            # smallest_dist and graph_for_neighbor_map now contain the smallest N distances between target_graph
            # and the training set.
            print "final smallest_dist: ", smallest_dist
            print "final map: ", graph_for_neighbor_map

            neighbor_classes = [self.id_to_class_map[id] for id in graph_for_neighbor_map.values()]
            modal_class = weighted_mode(neighbor_classes, weights)
            print "most common class for target graph: %s is %s" % (target_graph, modal_class)
            # if there is a tie, break it randomly
            if len(modal_class) > 1:
                modal_class = np.random.choice(modal_class, size=1)
                print "breaking tie by choosing class: %s" % modal_class
            y_pred.append(modal_class)

        return np.asarray(y_pred)


    ######## Private Methods #######

    def _graph_spectral_similarity(self, g1, g2, t = 0.9):
        """
        Returns the eigenvector similarity, between [0, 1], for two NetworkX graph objects, as
        the sum of squared differences between the sets of Laplacian matrix eigenvalues that account
        for a given fraction of the total sum of the eigenvalues (default = 90%).

        Similarity scores of 0.0 indicate identical graphs (given the adjacency matrix, not necessarily
        node identity or annotations), and large scores indicate strong dissimilarity.  The statistic is
        unbounded above.
        """
        l1 = nx.spectrum.laplacian_spectrum(g1, weight=None)
        l2 = nx.spectrum.laplacian_spectrum(g2, weight=None)
        k1 = self._get_num_eigenvalues_sum_to_threshold(l1, threshold=t)
        k2 = self._get_num_eigenvalues_sum_to_threshold(l2, threshold=t)
        k = min(k1,k2)
        sim = sum((l1[:k] - l2[:k]) ** 2)
        return sim

    def _get_num_eigenvalues_sum_to_threshold(self, spectrum, threshold = 0.9):
        """
        Given a spectrum of eigenvalues, find the smallest number of eigenvalues (k)
        such that the sum of the k largest eigenvalues of the spectrum
        constitutes at least a fraction (threshold, default = 0.9) of the sum of all the eigenvalues.
        """
        if threshold is None:
            return len(spectrum)

        total = sum(spectrum)
        if total == 0.0:
            return len(spectrum)

        spectrum = sorted(spectrum, reverse=True)
        running_total = 0.0

        for i in range(len(spectrum)):
            running_total += spectrum[i]
            if running_total / total >= threshold:
                return i + 1
        # guard
        return len(spectrum)
