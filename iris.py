
# coding: utf-8

# # Iris Dataset Classification using Gradient Boosting & Single-Layer Percepton

# We use Gradient Boosting Classifier using XGBoost and a Single-Layer Perceptron to train the Iris dataset.

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


path = 'iris.data'


cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(path, delimiter=' ', index_col=False, header=None, 
                 names=cols)


# We convert the values to 16-bit floating values to reduce memory usage, and accuracy.


f64_cols = df.select_dtypes(['float64']).columns
df[f64_cols] = df[f64_cols].astype('float16')


# Normalize the dataset to improve training performance and accuracy.


sc = StandardScaler()
sc.fit(df[cols[:-1]])
df[cols[:-1]] = sc.transform(df[cols[:-1]])
print(df.head())


# Plotting all features among themselves clearly suggests that some features discriminate the data very well. A Decision Tree algorithm is the most suitable one for such data.


sns.pairplot(df, hue='species', x_vars=cols[:-1], y_vars=cols[:-1])


# Convert the labels into numerical categories.


df['species'], labels = pd.factorize(df['species'])


X = df[cols[:-1]].as_matrix()
y = df['species'].as_matrix()
print(X.shape, y.shape)


# Split the dataset into Training and Test dataset.

# In[363]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, 
                                                    shuffle=True)
print((X_train.shape, y_train.shape), (X_test.shape, y_test.shape))


# We classify our data using the XGBoost classifier. We use Grid Search followed by StratifiedKFold for cross-validation. 
# We use the best model using the best parameters found by Grid Search.


params = {'max_depth': [4, 6, 8], 
          'n_estimators': [10, 25, 50]}
clf = XGBClassifier()
cross_validation = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(clf, scoring='accuracy', 
                           param_grid=params, 
                           cv=cross_validation)
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)


print("XGBoost Score = {0:.2f}%".format(100. * best_model.score(X_test, y_test)))


# We use a Single-Layer Perceptron with Softmax activation and backpropagation using RMSProp to train the classifier.


model = Sequential()
model.add(Dense(3, activation='softmax', input_shape=(4, )))
model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', metrics=['accuracy'])
_y_train = np_utils.to_categorical(y_train, 3)
_y_test = np_utils.to_categorical(y_test, 3)
model.fit(X_train, _y_train, batch_size=10, epochs=200, verbose=0)


loss, score = model.evaluate(X_test, _y_test)
print("Single-Layer Perceptron\nLoss = {0:.2f}\nScore = {1:.2f}%".format(loss, 100. * score))

