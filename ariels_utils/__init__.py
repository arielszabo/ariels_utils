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
from sklearn.model_selection import StratifiedKFold, train_test_split
import multiprocessing
import pandas as pd

class test_model(object):
    def __init__(self, model, x, y, scoring_method, shuffle=True, stratify=None, k_folds=None, n_jobs=1, random_state=None):
        self.model = model
        self.x = x
        self.y = y
        self.scoring_method = scoring_method
        # -1 indicates we want all the cores and if we pass multiprocessing.Pool() a None it will use all the cores:
        self.n_jobs = None if n_jobs == -1 else n_jobs
        self.k_folds = k_folds
        self.shuffle = shuffle
        self.stratify = stratify
        self.random_state = random_state
        # todo: ask dad if its good to use getattr ?

    def single_model_run(self, x_train, x_test, y_train, y_test):
        self.model.fit(x_train, y_train)

        train_score = metrics.get_scorer(self.scoring_method)(self.model, x_train, y_train)
        test_score = metrics.get_scorer(self.scoring_method)(self.model, x_test, y_test)

        return train_score, test_score

    def split_x_and_y(self):
        split_data_and_target = []
        if self.k_folds:  # if it's not None
            for train_index, test_index in StratifiedKFold(n_splits=self.k_folds, shuffle=self.shuffle,
                                                           random_state=self.random_state).split(self.x, self.y):
                x_train, x_test = self.x.iloc[train_index], self.x.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

                split_data_and_target.append([x_train, x_test, y_train, y_test])

        else:
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, random_state=self.random_state,
                                                                stratify=self.stratify, shuffle=self.shuffle)
            split_data_and_target.append([x_train, x_test, y_train, y_test])

        return split_data_and_target

    def run(self):
        input_to_multi = self.split_x_and_y()
        with multiprocessing.Pool(self.n_jobs) as p:
            results = p.starmap(self.single_model_run, input_to_multi)

        return pd.DataFrame(results,
                            columns=[f'train_{self.scoring_method}', f'test_{self.scoring_method}']).mean(axis=0)