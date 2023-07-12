# Combining transformers and estimators in a pipeline
# =============================================================================
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ch6_Loading_the_Breast_Cancer_Wisconsin_dataset import X_train, \
    y_train, X_test, y_test

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs')
                        )

if __name__ == '__main__':
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    test_accuracy = pipe_lr.score(X_test, y_test)
    print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
