
# coding: utf-8

# In[3]:

import sklearn.datasets as skd
import pandas as pd

# generate a problem with more overlap
X, Y = skd.make_classification(n_samples=10000, n_classes=10, n_informative = 5)
df_x = pd.DataFrame(X)
df_y = pd.DataFrame(Y)
df_x.to_csv("classification-10k-10classes-x.csv", index=False)
df_y.to_csv("classification-10k-10classes-y.csv", index=False)


# In[ ]:



