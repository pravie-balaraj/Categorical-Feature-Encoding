
# coding: utf-8

# In[30]:


from sklearn.model_selection import train_test_split
import keras
from keras.layers import BatchNormalization
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import keras
import mlflow


# In[31]:


train = pd.read_csv(train.csv")
test = pd.read_csv(test.csv")
X=train.drop(['target'],axis=1)
y=train['target']
labels = train.pop('target')
train_id = train.pop("id")
test_id = test.pop("id")


# In[32]:


print(train.shape)
print(test.shape)


# In[33]:


x=y.value_counts()
plt.bar(x.index,x)
plt.gca().set_xticks([0,1])
plt.title('distribution of target variable')
plt.show()


# In[34]:


labels = labels.values


# In[35]:


data = pd.concat([train, test])
data["ord_5a"] = data["ord_5"].str[0]
data["ord_5b"] = data["ord_5"].str[1]
data.drop(["bin_0", "ord_5"], axis=1, inplace=True)
columns = [i for i in data.columns]
dummies = pd.get_dummies(data,
                         columns=columns,
                         drop_first=True,
                         sparse=True)

del data


# In[36]:


dummies = np.array(dummies)
train = dummies[:train.shape[0], :]
test = dummies[train.shape[0]:, :]

del dummies


# In[40]:


with mlflow.start_run():
    model = Sequential()

    model.add(Dense(units=128, activation='relu', input_dim=train.shape[1]))

    model.add(
        BatchNormalization(momentum=0.17,
                       epsilon=1e-5,
                       gamma_initializer="uniform"))

    model.add(Dropout(0.15))
    model.add(Dense(units=64, activation='softmax'))
    model.add(Dropout(0.05))
    model.add(Dense(units=2, activation='softmax'))



    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.01),
                  metrics=['accuracy'])

    model.fit(train, labels, epochs=1, batch_size=64)

    
    #mlflow.log_metric("loss", loss)


# In[41]:


import mlflow.sklearn
a = []

a = model.predict(test)[:, 1]

mlflow.sklearn.log_model(a, "model")


#submission["target"] = submission["target"].astype(float)


# In[11]:


print(a, test_id)
print(a.shape, test_id.shape)


# In[12]:


submission = pd.DataFrame({'id': test_id, 'target': a})

submission.to_csv(r"C:\Users\pravi\OneDrive\Desktop\kaggle\submission.csv", index=False)


# In[13]:


submission.head()

