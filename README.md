# sklearn-mmadsen

Additional classes and tools for ML and statistics built on top of scikit-learn, numpy, scipy, and various DNN libraries

I add code to this repository when I've got pieces that are suitably reusable for many projects.  Usually there's a life
cycle where a bit of special purpose code exists in an analysis script, and then it gets cleaned up into a class, then
it gets the full scikit-learn treatment when I want to use it in Pipelines.  Once it gets to the latter stage, I'll put 
those pieces of code here.  

## Deep Neural Network classes

* ParameterizedDNNClassifier - simple Keras fully-connected deep classifier builder, following scikit-learn API


## Graph Classes

Measuring graph similiarity is the subject of much research in fields with rooted and labeled graphs and trees, but few
algorithms address situations with unrooted and unlabeled trees.  Edit distance algorithms tend to rely upon vertex id's
to analyze what vertices are changed between two graphs, and various other algorithms (e.g., Zhang/Shasha) rely upon 
the ordering of nodes and edges in addition to having clear vertex id's.  

Without vertex id's, similarity measures have only the pattern of edge connections to rely upon, which leads to finding 
similarity by manipulations of the adjacency or Laplacian matrices of the graph.  Treating the spectrum of eigenvalues
from the Laplacian of a graph as a vector, a simple measure of the edge similarity between two graphs is simply the 
squared differences (i.e., Euclidean distance) between spectra.  This algorithm is easily implemented in the form of a 
nearest neighbor search between target graphs and a training set, and we can augment the training set with class labels, 
and form a classifier (see the `GraphEigenvalueNearestNeighbors` class for details).  