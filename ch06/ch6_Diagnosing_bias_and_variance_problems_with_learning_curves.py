# Diagnosing bias and variance problems with learning curves
# ---------------------------------------------------------------------------------
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ch6_Loading_the_Breast_Cancer_Wisconsin_dataset import X_train, y_train


def learning_curves_func():
    # Create a pipeline that standardizes the data, then creates a model
    pipe_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty='l2', random_state=1,
                           solver='lbfgs', max_iter=10000)
    )

    # Calculate the training and test scores for the learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipe_lr,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=10,
        n_jobs=-1,
        random_state=1
    )

    # Calculate the mean and standard deviation of the training scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate the mean and standard deviation of the test scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    import matplotlib.pyplot as plt

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    learning_curves_func()
   
