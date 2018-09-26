# This is a self made utils package for many use cases
from sklearn import metrics
import multiprocessing
import pandas as pd


class IteratorReStarter(object):
    def __init__(self, iterator_object_instance, iterator_input_params={}):
        self.iterator_input_params = iterator_input_params
        self.iterator_object_instance = iterator_object_instance

    def __iter__(self):
        if isinstance(self.iterator_input_params, list):
            return self.iterator_object_instance(*self.iterator_input_params)
        elif isinstance(self.iterator_input_params, dict):
            return self.iterator_object_instance(**self.iterator_input_params)


# todo: add documentation

class MLTester(object):
    def __init__(self, estimator, x, y, scoring_method, n_jobs=1, splitting_method=None, splitting_method_params=None, groups=None):
        """

        :param estimator: estimator object

                    This is assumed to implement the scikit-learn estimator interface.

        :param x: array-like, shape (n_samples, n_features)

                    Training data, where n_samples is the number of samples and n_features is the number of features.

        :param y: array-like, shape (n_samples,)

                    The target variable for supervised learning problems.

        :param scoring_method: string

                    A single string to evaluate the predictions on the test set.

                    see http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for more info.

        :param n_jobs: int, default=1

                    Number of jobs to run in parallel.

        :param splitting_method: callable

                    A splitting function from scikit-learn's model_selection sub-module.

                    Like LeavePOut, KFold, TimeSeriesSplit etc ...

        :param splitting_method_params: dict

                    Parameters for the splitting_methods provided function.

                    Like n_splits, max_train_size etc ...

        :param groups: array-like, with shape (n_samples,), optional, for some splitting_methods it's ignored.

                    Group labels for the samples used while splitting the dataset into train/test set.
        """
        self.estimator = estimator
        self.x = x
        self.y = y
        self.scoring_method = scoring_method
        # -1 indicates we want all the cores and if we pass multiprocessing.Pool() a None it will use all the cores:
        self.n_jobs = None if n_jobs == -1 else n_jobs
        self.splitting_method = splitting_method
        self.splitting_method_params = splitting_method_params
        self.groups = groups

    def single_model_run(self, x_train, x_test, y_train, y_test):
        """
        Fit the estimator and get the score (based on the scoring method given) on the train and test data.

        :param x_train: array-like, shape (n_samples, n_features)

                    Training data for fitting the model / estimator).

                    where n_samples is the number of samples and n_features is the number of features.

        :param x_test:array-like, shape (n_samples, n_features)

                    Training data for testing the model / estimator).

                    where n_samples is the number of samples and n_features is the number of features.

        :param y_train: array-like, shape (n_samples,)

                The target variable for fitting the model / estimator

        :param y_test: array-like, shape (n_samples,)

                The target variable for testing the model / estimator.

        :return: Tuple
                (train score, test score)
        """
        self.estimator.fit(x_train, y_train)

        train_score = metrics.get_scorer(self.scoring_method)(self.estimator, x_train, y_train)
        test_score = metrics.get_scorer(self.scoring_method)(self.estimator, x_test, y_test)

        return train_score, test_score

    def _split_x_and_y(self):
        """
        Splits the provided x and y data by the provided splitting_method.

        :return: List containing train-test split of inputs.
        """
        split_data_and_target = []
        for train_index, test_index in self.splitting_method(**self.splitting_method_params).split(self.x, self.y,
                                                                                                   self.groups):
            x_train, x_test = self.x.iloc[train_index], self.x.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            split_data_and_target.append([x_train, x_test, y_train, y_test])

        return split_data_and_target

    def run(self):
        """
        Run the full testing.
        Split the given X and y based on the splitting_method.
        Fit and estimate your given model in parallel (n_jobs).

        :return: a Pandas' DataFrame
        """
        input_to_multi = self._split_x_and_y()
        with multiprocessing.Pool(self.n_jobs) as p:
            results = p.starmap(self.single_model_run, input_to_multi)

        return pd.DataFrame(results,
                            columns=[f'train_{self.scoring_method}', f'test_{self.scoring_method}']).describe()

# todo: Stemmer and toknizer + word embedding 2 features
