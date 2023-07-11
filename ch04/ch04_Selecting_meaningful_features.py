# Read wine dataset by using pandas
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())

# Split the dataset into training and testing datasets
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# Split the dataset into 70% training and 30% testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=0,
    stratify=y)

# Standardize the features by using StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Standardize the features by using RobustScaler

rs: RobustScaler = RobustScaler()
X_train_rs: np.ndarray = rs.fit_transform(X_train)
X_test_rs: np.ndarray = rs.transform(X_test)

# Use logistic regression classifier to train the model on the training dataset
logres: LogisticRegression = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    multi_class='ovr',
)
# fit the model
logres.fit(X_train_rs, y_train)
# print accuracy score
print('Training accuracy:', logres.score(X_train_rs, y_train))
print('Test accuracy:', logres.score(X_test_rs, y_test))

intercept = logres.intercept_
coef = logres.coef_
print('intercept:', intercept)
print('coef:', coef)

# Increase the regularization strength and check the accuracy score again
logres2: LogisticRegression = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    multi_class='ovr',
    C=0.1
)

# fit the model
logres2.fit(X_train_rs, y_train)
# print accuracy score
print('Training accuracy:', logres2.score(X_train_rs, y_train))
print('Test accuracy:', logres2.score(X_test_rs, y_test))

intercept = logres2.intercept_
coef = logres2.coef_
print('intercept:', intercept)
print('coef:', coef)
