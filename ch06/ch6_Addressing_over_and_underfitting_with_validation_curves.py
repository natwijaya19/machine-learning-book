# Addressing over- and underfitting with validation curves
# ------------------------------------------------------------------------------------
from matplotlib import pyplot as plt
from sklearn.model_selection import validation_curve

from ch6_Combining_transformers_and_estimators_in_pipeline import pipe_lr
from ch6_Loading_the_Breast_Cancer_Wisconsin_dataset import X_train, y_train


def plot_validation_curve():
    # Calculate the training and test scores for varying parameter values
    praram_range: list = [10 ** c for c in range(-5, 6)]

    train_scores, test_scores = validation_curve(
        estimator=pipe_lr,
        X=X_train,
        y=y_train,
        param_name='logisticregression__C',
        param_range=praram_range,
        cv=10
    )

    # Calculate the mean and standard deviation of the training and test scores
    # for each parameter value
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    # plot the validation curves
    plt.plot(praram_range, train_mean, color='blue', marker='o', markersize=5,
             label='training accuracy')

    plt.fill_between(praram_range, train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(praram_range, test_mean, color='green', linestyle='--', marker='s',
             markersize=5, label='validation accuracy')

    plt.fill_between(praram_range, test_mean + test_std, test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.show()


if __name__ == '__main__':
    plot_validation_curve()
