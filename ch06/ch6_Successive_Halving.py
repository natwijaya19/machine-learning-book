# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingGridSearchCV
#
# from ch6_Loading_the_Breast_Cancer_Wisconsin_dataset import y_test, X_test, \
#     y_train, X_train
# from ch6_Tuning_hyperparameters_via_grid_search import pipe_svc
#
# param_range = [
#     10 ** i for i in range(-3, 5)
# ]
# param_grid = [
#     {"svc__C": param_range, "svc__kernel": ["linear"]},
#     {"svc__C": param_range, "svc__gamma": param_range, "svc__kernel": ["rbf"]}
# ]
#
# hs = HalvingGridSearchCV(
#     estimator=pipe_svc,
#     param_distribution=param_grid,
#     n_candidates='exhaust',
#     factor=1.5,
#     resource='n_samples',
#     random_state=1,
#     n_jobs=-1,
# )
#
# hs = hs.fit(X_train, y_train)
# print(f"hs.best_score_: {hs.best_score_}")
#
# clf = hs.best_estimator_
# print(f"clf.score(X_test, y_test): {clf.score(X_test, y_test)}")
