import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from charityML import split_data
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, fbeta_score


# where to find possible scoring values?
# http://scikit-learn.org/stable/modules/model_evaluation.html
def plot_validation_curve(sample=None,
                          param_name=None,
                          scoring='accuracy',
                          param_range=None):
    train_mean, test_mean = train_model(sample, param_name, scoring, param_range)
    plt.plot(param_range, train_mean, 'o-', color='r', label='train')
    plt.plot(param_range, test_mean, 'o-', color='g', label='test')
    format_plot(sample, param_name, scoring)
    plt.show()


def train_model(sample, param_name, scoring, param_range):
    model = DecisionTreeClassifier(random_state=42)
    train_scores, test_scores = validation_curve(
        model, X_train, y_train,
        cv=10,
        param_name=param_name, param_range=param_range,
        scoring=scoring)

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    return train_mean, test_mean


def format_plot(sample, param_name, scoring):
    plt.xlabel(param_name)
    plt.ylabel('mean score')
    plt.legend()
    plt.title(f'validation curve: scoring=\'{scoring}\'; \nsample={sample*100:.0f}%')


if __name__ == '__main__':
    sample = .1
    X_train, X_test, y_train, y_test = split_data(sample=sample)
    scorer = make_scorer(fbeta_score, beta=0.5, average='micro')

    # plot_validation_curve(sample=sample,
    #                       param_name='max_depth',
    #                       param_range=np.arange(1, 11),
    #                       scoring=scorer)

    min_samples_range = np.array([5, 10, 25, 50, 100, 200, 300, 400, 500, 1000])
    plot_validation_curve(sample=sample,
                          param_name='min_samples_split',
                          param_range=min_samples_range,
                          scoring=scorer)
