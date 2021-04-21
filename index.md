
# Question 4
### In the lab, based on the test dataset, which model predicted the most false negatives? Use a max depth of 5 for your trees, and the random state = 1693. For the random forest classifier, n_estimators should be 1000.


```markdown

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as CM
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

data = pd.read_csv("drive/MyDrive/Colab Notebooks/example_data_classification.csv", header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
admitted = data.loc[y == 1]
not_admitted = data.loc[y == 0]
df = pd.DataFrame(data=data.values,columns=['Exam 1','Exam 2','Status'])

# Classification Tree
from sklearn import tree
dtree = tree.DecisionTreeClassifier(max_depth=5, random_state=1693)
dtree.fit(X, y)
predicted_classes_dtree = dtree.predict(X)

spc = ['Not Admitted','Admitted']
cm = CM(y,predicted_classes_dtree)
pd.DataFrame(cm, columns=spc, index=spc)

![Screen Shot 2021-04-21 at 10 10 57 AM](https://user-images.githubusercontent.com/78621124/115568041-e4eeb300-a289-11eb-9d7c-d0158aa94001.png)

# Naive Bays
from sklearn.naive_bayes import GaussianNB
gauss = GaussianNB()
gauss.fit(X, y);
predicted_classes_gauss = gauss.predict(X)

spc = ['Not Admitted','Admitted']
cm = CM(y,predicted_classes_gauss)
pd.DataFrame(cm, columns=spc, index=spc)

![Screen Shot 2021-04-21 at 10 11 40 AM](https://user-images.githubusercontent.com/78621124/115568120-fdf76400-a289-11eb-9d63-a00abbc72688.png)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rforest = RandomForestClassifier(random_state=1693, max_depth=5, n_estimators = 1000)
rforest.fit(X, y);
predicted_classes_rforest = rforest.predict(X)

spc = ['Not Admitted','Admitted']
cm = CM(y,predicted_classes_rforest)
pd.DataFrame(cm, columns=spc, index=spc)

![Screen Shot 2021-04-21 at 10 12 14 AM](https://user-images.githubusercontent.com/78621124/115568191-10719d80-a28a-11eb-98d2-141995775c43.png)

```
### I couldn't find anything about what a Bayesian Tree is

## Answer: Naive Bays has the most false negatives


# Question 14
### If we retain only two input features, such as "mean radius" and "mean texture" and apply the Gaussian Naive Bayes model for classification, then the average accuracy determined on a 10-fold cross validation with random_state = 1693 is (do not use the % notation, just copy the first 4 decimals)

```markdown

from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB

data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)

y = data.target
x = data.data

feats = ['mean radius', 'mean texture']
X = df[feats].values

from sklearn.pipeline import Pipeline
model = GaussianNB()
scale = StandardScaler()
pipe = Pipeline([('scale',scale),('Classifier',model)])

kf = KFold(n_splits=10, random_state=1693,shuffle=True)
def DoKFold_scaled(X,y,model):
  accuracy = []
  pipe = Pipeline([('scale',scale), ('Classifier',model)])
  for idxtrain, idxtest in kf.split(X):
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    predicted_classes = pipe.predict(Xtest)
    accuracy.append(accuracy_score(ytest,predicted_classes))
  return (np.mean(accuracy))
  
model = GaussianNB()
DoKFold_scaled(X,y,model)
```

## Answer = 0.8805


# Question 15
### From the data retain only two input features, such as "mean radius" and "mean texture" and apply the Random Froest model for classification with 100 trees, max depth of 7 and random_state=1693; The average accuracy determined on a 10-fold cross validation with the same random state is (do not use the % notation, just copy the first 4 decimals)

```markdown
rforest = RandomForestClassifier(random_state=1693, max_depth=7, n_estimators = 100)
DoKFold_scaled(sub,y,rforest)
```

## Answer = 0.8822


# Question 16

From the data retain only two input features, such as "mean radius" and "mean texture" we build an Artificial Neural Network for classification that has three hidden layers with 16, 8 and 4 neurons respectively.

Assume that the neurons in the hidden layer have the rectified linear activation ('relu') and the kernel initializer uses the random normal distribution. Assume the output layer has only one neuron with 'sigmoid' activation.

You will compile the model with

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

and fit the model with 150 epochs and

validation_split=0.25,batch_size=10,shuffle=False

The average accuracy determined on a 10-fold cross validation (random state=1693) is closer to

```markdown
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acc

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y)
scale = StandardScaler()

model = Sequential()
model.add(Dense(16,kernel_initializer='random_normal', input_dim=2, activation='relu'))
model.add(Dense(8,kernel_initializer='random_normal', activation='relu'))
model.add(Dense(4,kernel_initializer='random_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid')) # the logistic sigmoid: 1/(1+e^(-x))
model.compile(loss='binary_crossentropy', optimizer='adam', metr

AC = []
for idxtrain, idxtest in kf.split(X):
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    Xstrain = scale.fit_transform(Xtrain)
    Xstest = scale.transform(Xtest)
    model.fit(Xstrain, ytrain, epochs=150, shuffle=False, validation_split=0.25,        batch_size=10, verbose=0)
    Ac.append(acc(ytest,model.predict_classes(Xstest))
    
np.mean(AC)
```

## Answer = 0.906

