# Load wine dataset
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
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

# LDA via scikit-learn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

# train logistic regression classifier on the lower-dimensional dataset
lr = LogisticRegression(
    multi_class='ovr', random_state=1, solver='lbfgs'
)
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(
    X=X_train_lda,
    y=y_train,
    classifier=lr,
)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# test data
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(
    X=X_test_lda,
    y=y_test,
    classifier=lr,
)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# print accuracy score of the logistic regression classifier on the training and test dataset
from sklearn.metrics import accuracy_score

y_train_pred = lr.predict(X_train_lda)
print('Accuracy (train): %.2f' % accuracy_score(y_train, y_train_pred))
y_test_pred = lr.predict(X_test_lda)
print('Accuracy (test): %.2f' % accuracy_score(y_test, y_test_pred))
