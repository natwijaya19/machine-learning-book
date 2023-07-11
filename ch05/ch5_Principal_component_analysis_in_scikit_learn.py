# Load wine dataset
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from plot_helper import plot_decision_regions


wine = load_wine()
X = wine.data
y = wine.target
print(X.shape)

# split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# standardize the features

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

pca = PCA(n_components=2)
logreg = LogisticRegression(
    multi_class='ovr', random_state=1, solver='lbfgs')

# dimensionality reduction
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# fitting the logistic regression model on the reduced dataset
logreg.fit(X_train_pca, y_train)

# plot decision regions
plot_decision_regions(
    X=X_train_pca,
    y=y_train,
    classifier=logreg)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# print the accuracy of the model on the training and test sets
print('Training accuracy:', logreg.score(X_train_pca, y_train))
print('Test accuracy:', logreg.score(X_test_pca, y_test))
