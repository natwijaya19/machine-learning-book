# Load wine dataset
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

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

# construct the covariance matrix

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

# # plot the variance explained ratios of the eigenvalues
# import matplotlib.pyplot as plt
#
# tot = sum(eigen_vals)
# var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)
#
# plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
#         label='individual explained variance')
# plt.step(range(1, 14), cum_var_exp, where='mid',
#             label='cumulative explained variance')
