# Loading the Breast Cancer Wisconsin dataset
# ---------------------------------------------------------------------------------

# Obtaining the Breast Cancer Wisconsin dataset by using pd.read_csv
import pandas as pd

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
    header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values

# Encoding the class labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into a separate training and test dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.20,
                     stratify=y,
                     random_state=1)

if __name__ == '__main__':
    print(le.classes_)
