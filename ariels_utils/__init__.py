# This is a self made utils package for many use cases


class IteratorReStarter(object):
    def __init__(self, iterator_object_instance, iterator_input_params):
        self.iterator_input_params = iterator_input_params
        self.iterator_object_instance = iterator_object_instance

    def __iter__(self):
        if isinstance(self.iterator_input_params, list):
            return self.iterator_object_instance(*self.iterator_input_params)
        elif isinstance(self.iterator_input_params, dict):
            return self.iterator_object_instance(**self.iterator_input_params)

# todo: add my scoring func - DONE !
# todo: determine number of cores to run on - DONE !
# todo: modelling Kfold CV with random vs stratified sampling and regression and classification selection etc.
# todo: add my split of data: can be traditional train_test_split and can be Kfold (if so fold needs a number)
# todo: determine if Stratified or not
# todo: determine if shuffle or repeated or not

# todo: work both for regression and classification
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split, RepeatedStratifiedKFold, RepeatedKFold
import multiprocessing
import pandas as pd

class test_model(object):
    def __init__(self, model, x, y, scoring_method, n_jobs=1, splitting_method=None, splitting_method_params=None):
        self.model = model
        self.x = x
        self.y = y
        self.scoring_method = scoring_method
        # -1 indicates we want all the cores and if we pass multiprocessing.Pool() a None it will use all the cores:
        self.n_jobs = None if n_jobs == -1 else n_jobs
        self.splitting_method = splitting_method
        self.splitting_method_params = splitting_method_params

    def single_model_run(self, x_train, x_test, y_train, y_test):
        self.model.fit(x_train, y_train)

        train_score = metrics.get_scorer(self.scoring_method)(self.model, x_train, y_train)
        test_score = metrics.get_scorer(self.scoring_method)(self.model, x_test, y_test)

        return train_score, test_score

    def _split_x_and_y(self):
        split_data_and_target = []
        for train_index, test_index in self.splitting_method(**self.splitting_method_params).split(self.x, self.y):
            x_train, x_test = self.x.iloc[train_index], self.x.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            split_data_and_target.append([x_train, x_test, y_train, y_test])

        return split_data_and_target

    def run(self):
        input_to_multi = self._split_x_and_y()
        with multiprocessing.Pool(self.n_jobs) as p:
            results = p.starmap(self.single_model_run, input_to_multi)

        return pd.DataFrame(results,
                            columns=[f'train_{self.scoring_method}', f'test_{self.scoring_method}']).mean(axis=0)

# todo: Stemmer and toknizer + word embedding 2 features
