# K-fold cross-validation
import numpy as np
from sklearn.model_selection import cross_val_score

from ch6_Combining_transformers_and_estimators_in_pipeline import pipe_lr
from ch6_Loading_the_Breast_Cancer_Wisconsin_dataset import X_train, \
    y_train

if __name__ == '__main__':
    scores = cross_val_score(
        estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=4
    )

    print('CV accuracy scores: %s' % scores)

    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
