
# coding: utf-8

# ## Test script for using ParameterizedDNNClassifier with SKL GridSearchCV

# Keras is a library for quickly creating deep neural network (DNN) models out of standard layer types, activation functions, and other components without hand-crafting Theano or other low level code.  Keras has the additional advantage that its models can be run with Theano and TensorFlow backends without modification (usually, unless you're writing your own Keras extensions).  
# 
# Since one of the most common tasks I do with ML tools is supervised multiclass classification, I wanted an easy way to include a DNN for such classifiers whenever I'm screening models in scikit-learn.  ParameterizedDNNClassifier takes some simple parameters and the generates an appropriate multilayer DNN in Keras, with one or more fully-connected hidden layers (with specifiable activation function, defaulting to ReLU), and configurable input and output layers.  Future additions will make the optimizer and other aspects parameterized as well.  
# 
# ParameterizedDNNClassifier subclasses BaseEstimator and ClassifierMixin from scikit-learn, and provides appropriate score and predict functions which allow it to act like any other SKL estimator.  With one exception (getting training history from the underlying Keras object), you can use ParameterizedDNNClassifier in a Pipeline or in GridSearchCV, etc.  
# 
# The following test harness does a simple grid search cross validation over a synthetic classification data set with 10K data points and 10 classes.  
# 

# In[1]:

import random
import numpy as np 
import pandas as pd 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import pprint as pp

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn_mmadsen import ParameterizedDNNClassifier

get_ipython().magic(u'matplotlib inline')


# In[2]:

## Seaborn confusion matrix heatmap
def confusion_heatmap(y_test, y_pred, labels):
    from sklearn.metrics import confusion_matrix
    mat = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                     xticklabels=labels, yticklabels=labels)


# In[3]:

###### Replication #######

#random.seed(7112)


# In[4]:

df_x = pd.read_csv("../testdata/classification-10k-10classes-x.csv.gz")
df_y = pd.read_csv("../testdata/classification-10k-10classes-y.csv.gz")


# In[5]:

############ prepare data ###########

# specify the correct data types because we're probably using the GPU
X = df_x.astype(np.float32)
y = df_y.astype(np.int32)

# one-hot encode the class label since the output layer of the DNN will have multiple units, 
# each corresponding to a class
y = pd.concat([y, pd.get_dummies(y['0']).rename(columns=lambda x: 'col_' + str(x))], axis=1)
y.drop('0', axis=1, inplace=True	)

# get pure numpy arrays
X = X.values
y = y.values

### create a train/test split ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

x_train_rows = X_train.shape[0]
x_train_cols = X_train.shape[1]
x_test_rows = X_test.shape[0]
x_test_cols = X_test.shape[1]
y_test_cols = y_test.shape[1]

# make sure the data arrays are the correct shape, or Theano will never let you hear the end of it...
X_train = X_train.reshape(x_train_rows, x_train_cols)
X_test = X_test.reshape(x_test_rows, x_test_cols)

print "Prepared data sets"
print "Train:"
print "X_train: ", X_train.shape
print "y_train: ", y_train.shape
print "\nTest:"
print "X_test: ",X_test.shape
print "y_test: ",y_test.shape 


# In[6]:

params = {
	'clf__dropout_fraction': [0.9, 0.5],
	'clf__sgd_lr': [0.01, 0.1],
}

est = ParameterizedDNNClassifier(input_dimension=20,
        output_dimension=10,
		num_dense_hidden=2,
		epochs=2,
		hidden_sizes=[1000,2000,1000],
        verbose=0)

grid_search = GridSearchCV(est, params, n_jobs = 1, verbose = 1)
grid_search.fit(X_train, y_train)

# This is Keras-specific code.  The fitted model stores the history of training 
# and validation accuracy so we can examine over or underfitting.  It is also 
# useful for retrieving the actual number of training epochs in the case that
# we have early stopping activated.

history = grid_search.best_estimator_.get_history()
actual_epoch_count = len(history['acc'])


# In[7]:

print "============= Best Estimator from GridSearchCV =============="

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters:")
best_params = grid_search.best_estimator_.get_params()
for param in sorted(best_params.keys()):
    print("param: %s: %r" % (param, best_params[param]))


# In[8]:

print "============== Evaluation on Holdout Test Set ============="

predictions = grid_search.predict(X_test)
actuals = np.argmax(y_test, axis=1)

print "accuracy on test: %s" % accuracy_score(actuals, predictions)

print(classification_report(actuals, predictions))


# In[9]:

# build a graph of the training/validation accuracy versus training epoch
# to look for overfitting

train_acc_hist = history['acc']
val_acc_hist = history['val_acc']
epoch_list = range(0, actual_epoch_count)

dat = { 'train_acc': train_acc_hist, 'val_acc': val_acc_hist }
hist_df = pd.DataFrame(data=dat, index=epoch_list)

plt.figure(figsize=(11,8.5), dpi=300)

plt.plot(hist_df.index, hist_df['train_acc'], color='green', linestyle='dashed', marker='+',
     markerfacecolor='black', markersize=7, label="Training Accuracy", alpha=0.4)
plt.plot(hist_df.index, hist_df['val_acc'], color='red', linestyle='dashed', marker='x',
     markerfacecolor='black', markersize=7, label="Validation Accuracy", alpha=0.4)
plt.legend(fontsize='large')

plt.xlabel('Epoch', fontsize='large')
plt.ylabel('Classification Accuracy', fontsize='large')
plt.title('Training and Validation Accuracy By Epoch', fontsize='large')
plt.show()


# In[10]:

labels = range(0, y_test_cols)
confusion_heatmap(actuals, predictions, labels)


# In[ ]:



