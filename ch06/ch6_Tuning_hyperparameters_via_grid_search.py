from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ch6_Loading_the_Breast_Cancer_Wisconsin_dataset import X_train, y_train, \
    X_test, y_test

# Grid search for tuning hyperparameters


# make_pipeline
pipe_svc = make_pipeline(
    StandardScaler(),
    SVC(random_state=1),
)

if __name__ == '__main__':

    praram_range: list = [10 ** i for i in range(-4, 5)]

    param_grid = [{
        'svc__C': praram_range,
        'svc__kernel': ['linear'],
    }, {
        'svc__C': praram_range,
        'svc__gamma': praram_range,
        'svc__kernel': ['rbf'],
    }]

    # Grid search for tuning hyperparameters
    gs: GridSearchCV = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring='accuracy',
        cv=10,
        n_jobs=-1,
    )

    # Fit the pipeline
    gs = gs.fit(X_train, y_train)
    print(f"gs.best_score_: {gs.best_score_}")
    print(f"gs.best_params_: {gs.best_params_}")
    print(f"gs.score_values_: {gs.score(X_test, y_test)}")

    # Use the best parameters to fit the data
    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    print(f"clf.score(X_test, y_test): {clf.score(X_test, y_test)}")
