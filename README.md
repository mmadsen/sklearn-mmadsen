# sklearn-mmadsen

Additional classes and tools for ML and statistics built on top of scikit-learn, numpy, scipy, and various DNN libraries

I add code to this repository when I've got pieces that are suitably reusable for many projects.  Usually there's a life
cycle where a bit of special purpose code exists in an analysis script, and then it gets cleaned up into a class, then
it gets the full scikit-learn treatment when I want to use it in Pipelines.  Once it gets to the latter stage, I'll put 
those pieces of code here.  

## Deep Neural Network classes

* ParameterizedDNNClassifier - simple Keras fully-connected deep classifier builder, following scikit-learn API



