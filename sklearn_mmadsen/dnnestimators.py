#!/usr/bin/env python

import numpy as np
import pprint as pp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.regularizers import WeightRegularizer
from keras.callbacks import EarlyStopping


class ParameterizedDNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Constructs a DNN classifier using the Keras library suitable for grid or random search
    cross-validation using scikit-learn scaffolding.  Defaults to a single hidden layer
    of ReLU activations, but is configurable to multiple layers with specifiable sizes,
    activation functions, dropout percentages, etc.

    The classifier subclasses scikit-learn's BaseEstimation and ClassifierMixin, and thus is
    ready to function in a cross-validation search or Pipeline for hyperparameter optimization.

    The classifier expects data in the following format:
    X:  numpy array with n columns of type np.float32
    Y:  numpy array with m columns, one-hot encoding the correct label for the corresponding row in X

    the output dimension of the classifier is thus the number of classes (and columns) in Y
    the input dimension of the classifier is thus the number of features (and columns) in X

    Because it treats target/labels as one-hot encoded (where most of scikit-learn's scoring functions
    except a single column with a class label instead), the class provides its own prediction and scoring
    functions that allow it to interoperate.

    The class has been tested with both Theano/GPU and TensorFlow backends, on CentOS 7.X Linux and OS X.

    Parameters:
    -----------

    input_dimension:  int, required.  Number of features in the dataset.
    output_dimension:  int, required.  Number of class labels the model is predicting
    dropout_fraction: float, required.  Fraction of input and dense layers to drop out to curb overfitting
    dense_activation: string, required.  Name of a Keras activation function to use on input and dense layers
    output_activation: string, required.  Name of a Keras activation function to use for classification on the output layer (usually softmax)
    num_dense_hidden: int, required.  Number of dense hidden layer/dropout pairs between input and output_activation
    hidden_sizes: array of int, required.  Gives the size of each hidden layer, and a final entry for the size of the input dimension to the output layer
    sgd_lr:  float, required.  Learning rate for the SGD optimizer.
    decay: float, required.  Decay rate for the learning rate during training.
    momentum: float, required.  Momentum value for Nesterov momentum gradient descent.
    epochs:  int, required.  Number of epochs to train unless early stopped.
    batch_size: int, required.  Size of mini batches to use during training.

    """

    def __init__(self,
                 input_dimension=10,
                 output_dimension=2,
                 dropout_fraction=0.5,
                 dense_activation='relu',
                 output_activation='softmax',
                 num_dense_hidden=1,
                 hidden_sizes=[5, 10],
                 sgd_lr=0.01,
                 decay=1e-6,
                 momentum=0.9,
                 epochs=100,
                 batch_size=500):

        self.input_dim = input_dimension
        self.output_dim = output_dimension
        self.drop_frac = dropout_fraction
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.num_dense = num_dense_hidden
        self.hidden_dim = hidden_sizes
        self.sgd_lr = sgd_lr
        self.sgd_decay = decay
        self.sgd_momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, x, y):
        model = Sequential()
        model.add(Dense(input_dim=self.input_dim, output_dim=self.hidden_dim[0],
                        init='glorot_uniform', activation=self.dense_activation))

        model.add(Dropout(self.drop_frac))

        for i in range(0, self.num_dense):
            j = i + 1
            model.add(Dense(input_dim=self.hidden_dim[i], output_dim=self.hidden_dim[i + 1],
                            init='glorot_uniform', activation=self.dense_activation))
            model.add(Dropout(self.drop_frac))

        model.add(Dense(input_dim=self.hidden_dim[self.num_dense], output_dim=self.output_dim,
                        activation=self.output_activation))

        # print model.summary()


        solver = SGD(lr=self.sgd_lr, decay=self.sgd_decay, momentum=self.sgd_momentum, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=solver)
        self.compiled_model = model
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.history = self.compiled_model.fit(x,
                                               y,
                                               nb_epoch=self.epochs,
                                               batch_size=self.batch_size,
                                               validation_split=0.1,
                                               verbose=1,
                                               show_accuracy=True,
                                               callbacks=[self.early_stopping])

    def predict(self, x):
        return self.compiled_model.predict_classes(x, batch_size=self.batch_size)

    def score(self, x, y):
        preds = self.compiled_model.predict_classes(x, batch_size=self.batch_size)
        actuals = np.argmax(y, axis=1)
        return accuracy_score(actuals, preds)

    def get_params(self, deep=True):
        return {
            'input_dimension': self.input_dim,
            'output_dimension': self.output_dim,
            'dropout_fraction': self.drop_frac,
            'dense_activation': self.dense_activation,
            'output_activation': self.output_activation,
            'num_dense_hidden': self.num_dense,
            'hidden_sizes': self.hidden_dim,
            'sgd_lr': self.sgd_lr,
            'decay': self.sgd_decay,
            'momentum': self.sgd_momentum,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_history(self):
        """
        Returns the fitting accuracy at each epoch.  This is a Keras-specific function,
        and thus isn't called through the scikit-learn API.  You will need to have access
        to the actual estimator object (not a Pipeline object) to call this method.
        """
        return self.history.history
