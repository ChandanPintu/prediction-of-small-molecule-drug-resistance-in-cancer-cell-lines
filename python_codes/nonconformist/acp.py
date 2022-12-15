#!/usr/bin/env python

"""
Aggregated conformal predictors
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.base import clone
from nonconformist.base import BaseEstimator
from nonconformist.util import calc_p


# -----------------------------------------------------------------------------
# Sampling strategies
# -----------------------------------------------------------------------------
class BootstrapSampler(object):

    def gen_samples(self, y, n_samples, problem_type):
        for i in range(n_samples):
            idx = np.array(range(y.size))
            train = np.random.choice(y.size, y.size, replace=True)
            cal_mask = np.array(np.ones(idx.size), dtype=bool)
            for j in train:
                cal_mask[j] = False
            cal = idx[cal_mask]

            yield train, cal


class CrossSampler(object):
    def gen_samples(self, y, n_samples, problem_type):
        if problem_type == 'classification':
            folds = StratifiedKFold(n_splits=n_samples)
            split_ = folds.split(np.zeros((y.size, 1)), y)
        else:
            folds = KFold(n_splits=n_samples)
            split_ = folds.split(np.zeros((y.size, 1)))

        for train, cal in split_:
            yield train, cal


class RandomSubSampler(object):

    def __init__(self, calibration_portion=0.3):
        self.cal_portion = calibration_portion

    def gen_samples(self, y, n_samples, problem_type):
        if problem_type == 'classification':
            splits = StratifiedShuffleSplit(
                n_splits=n_samples,
                test_size=self.cal_portion
            )

            split_ = splits.split(np.zeros((y.size, 1)), y)

        else:
            splits = ShuffleSplit(
                n_splits=n_samples,
                test_size=self.cal_portion
            )

            split_ = splits.split(np.zeros((y.size, 1)))

        for train, cal in split_:
            yield train, cal


# -----------------------------------------------------------------------------
# Conformal ensemble
# -----------------------------------------------------------------------------
class AggregatedCp(BaseEstimator):
    def __init__(self,
                 predictor,
                 sampler=BootstrapSampler(),
                 aggregation_func=None,
                 n_models=10):
        self.predictors = []
        self.n_models = n_models
        self.predictor = predictor
        self.sampler = sampler

        if aggregation_func is not None:
            self.agg_func = aggregation_func
        else:
            self.agg_func = lambda x: np.mean(x, axis=2)

    def fit(self, x, y):
        self.n_train = y.size
        self.predictors = []
        idx = np.random.permutation(y.size)
        x, y = x[idx, :], y[idx]
        problem_type = self.predictor.__class__.get_problem_type()
        samples = self.sampler.gen_samples(y,
                                           self.n_models,
                                           problem_type)
        for train, cal in samples:
            predictor = clone(self.predictor)
            predictor.fit(x[train, :], y[train])
            predictor.calibrate(x[cal, :], y[cal])
            self.predictors.append(predictor)

        if problem_type == 'classification':
            self.classes = self.predictors[0].classes

    def predict(self, x, significance=None):
        is_regression = \
            self.predictor.__class__.get_problem_type() == 'regression'

        n_examples = x.shape[0]

        if is_regression and significance is None:
            signs = np.arange(0.01, 1.0, 0.01)
            pred = np.zeros((n_examples, 2, signs.size))
            for i, s in enumerate(signs):
                predictions = np.dstack([p.predict(x, s)
                                         for p in self.predictors])
                predictions = self.agg_func(predictions)
                pred[:, :, i] = predictions
            return pred
        else:
            def f(p, x):
                return p.predict(x, significance if is_regression else None)

            predictions = np.dstack([f(p, x) for p in self.predictors])
            predictions = self.agg_func(predictions)

            if significance and not is_regression:
                return predictions >= significance
            else:
                return predictions


class CrossConformalClassifier(AggregatedCp):
    def __init__(self,
                 predictor,
                 n_models=10):
        super(CrossConformalClassifier, self).__init__(predictor,
                                                       CrossSampler(),
                                                       n_models)

    def predict(self, x, significance=None):
        ncal_ngt_neq = np.stack([p._get_stats(x) for p in self.predictors],
                                axis=3)
        ncal_ngt_neq = ncal_ngt_neq.sum(axis=3)

        p = calc_p(ncal_ngt_neq[:, :, 0],
                   ncal_ngt_neq[:, :, 1],
                   ncal_ngt_neq[:, :, 2],
                   smoothing=self.predictors[0].smoothing)

        if significance:
            return p > significance
        else:
            return p


class BootstrapConformalClassifier(AggregatedCp):
    def __init__(self,
                 predictor,
                 n_models=10):
        super(BootstrapConformalClassifier, self).__init__(predictor,
                                                           BootstrapSampler(),
                                                           n_models)

    def predict(self, x, significance=None):
        ncal_ngt_neq = np.stack([p._get_stats(x) for p in self.predictors],
                                axis=3)
        ncal_ngt_neq = ncal_ngt_neq.sum(axis=3)

        p = calc_p(ncal_ngt_neq[:, :, 0] + ncal_ngt_neq[:, :, 0] / self.n_train,
                   ncal_ngt_neq[:, :, 1] + ncal_ngt_neq[:, :, 0] / self.n_train,
                   ncal_ngt_neq[:, :, 2],
                   smoothing=self.predictors[0].smoothing)

        if significance:
            return p > significance
        else:
            return p
