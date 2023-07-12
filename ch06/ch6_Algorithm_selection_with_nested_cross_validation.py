# Algorithm selection with nested cross-validation
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score

from ch6_Loading_the_Breast_Cancer_Wisconsin_dataset import y_train, \
    X_train
from ch6_Tuning_hyperparameters_via_grid_search import pipe_svc


def nested_cv():
    param_range: list = [10 ** c for c in range(-4, 4)]
    param_grid: dict = [
        {"SVC__C": param_range,
         "SVC__kernel": ["linear"]},
        {"SVC__C": param_range,
         "SVC__gamma": param_range,
         "SVC__kernel": ["rbf"]}
    ]

    # GridSearchCV
    gs: GridSearchCV = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring="accuracy",
        cv=2,
        n_jobs=-1,
    )

    # Nested cross-validation
    scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)

    print(f"CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
#
# arr1 = np.arange(-4, 4)
# arr2 = arr1.shape
# print(arr2)

if __name__ == '__main__':
    nested_cv()
