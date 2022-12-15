#!/usr/bin/env python

"""
Inductive conformal predictors.
"""


from __future__ import division
from collections import defaultdict
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator

from nonconformist.base import RegressorMixin, ClassifierMixin
from nonconformist.util import calc_p


# -----------------------------------------------------------------------------
# Base inductive conformal predictor
# -----------------------------------------------------------------------------
class BaseIcp(BaseEstimator):

    def __init__(self, nc_function, condition=None):
        self.cal_x, self.cal_y = None, None
        self.nc_function = nc_function

        # Check if condition-parameter is the default function (i.e.,
        # lambda x: 0). This is so we can safely clone the object without
        # the clone accidentally having self.conditional = True.
        default_condition = lambda x: 0
        is_default = (callable(condition) and
                      (condition.__code__.co_code ==
                       default_condition.__code__.co_code))

        if is_default:
            self.condition = condition
            self.conditional = False
        elif callable(condition):
            self.condition = condition
            self.conditional = True
        else:
            self.condition = lambda x: 0
            self.conditional = False

    def fit(self, x, y):

        # TODO: incremental?
        self.nc_function.fit(x, y)

    def calibrate(self, x, y, increment=False):

        self._calibrate_hook(x, y, increment)
        self._update_calibration_set(x, y, increment)

        if self.conditional:
            category_map = np.array([self.condition((x[i, :], y[i]))
                                     for i in range(y.size)])
            self.categories = np.unique(category_map)
            self.cal_scores = defaultdict(partial(np.ndarray, 0))

            for cond in self.categories:
                idx = category_map == cond
                cal_scores = self.nc_function.score(self.cal_x[idx, :],
                                                    self.cal_y[idx])
                self.cal_scores[cond] = np.sort(cal_scores)[::-1]
        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score(self.cal_x, self.cal_y)
            self.cal_scores = {0: np.sort(cal_scores)[::-1]}

    def _calibrate_hook(self, x, y, increment):
        pass

    def _update_calibration_set(self, x, y, increment):
        if increment and self.cal_x is not None and self.cal_y is not None:
            self.cal_x = np.vstack([self.cal_x, x])
            self.cal_y = np.hstack([self.cal_y, y])
        else:
            self.cal_x, self.cal_y = x, y


# -----------------------------------------------------------------------------
# Inductive conformal classifier
# -----------------------------------------------------------------------------
class IcpClassifier(BaseIcp, ClassifierMixin):


    def __init__(self, nc_function, condition=None, smoothing=True):
        super(IcpClassifier, self).__init__(nc_function, condition)
        self.classes = None
        self.smoothing = smoothing

    def _calibrate_hook(self, x, y, increment=False):
        self._update_classes(y, increment)

    def _update_classes(self, y, increment):
        if self.classes is None or not increment:
            self.classes = np.unique(y)
        else:
            self.classes = np.unique(np.hstack([self.classes, y]))

    def predict(self, x, significance=None):

        # TODO: if x == self.last_x ...
        n_test_objects = x.shape[0]
        p = np.zeros((n_test_objects, self.classes.size))

        ncal_ngt_neq = self._get_stats(x)

        for i in range(len(self.classes)):
            for j in range(n_test_objects):
                p[j, i] = calc_p(ncal_ngt_neq[j, i, 0],
                                 ncal_ngt_neq[j, i, 1],
                                 ncal_ngt_neq[j, i, 2],
                                 self.smoothing)

        if significance is not None:
            return p > significance
        else:
            return p

    def _get_stats(self, x):
        n_test_objects = x.shape[0]
        ncal_ngt_neq = np.zeros((n_test_objects, self.classes.size, 3))
        for i, c in enumerate(self.classes):
            test_class = np.zeros(x.shape[0], dtype=self.classes.dtype)
            test_class.fill(c)

            # TODO: maybe calculate p-values using cython or similar
            # TODO: interpolated p-values

            # TODO: nc_function.calc_nc should take X * {y1, y2, ... ,yn}
            test_nc_scores = self.nc_function.score(x, test_class)
            for j, nc in enumerate(test_nc_scores):
                cal_scores = self.cal_scores[self.condition((x[j, :], c))][::-1]
                n_cal = cal_scores.size

                idx_left = np.searchsorted(cal_scores, nc, 'left')
                idx_right = np.searchsorted(cal_scores, nc, 'right')

                ncal_ngt_neq[j, i, 0] = n_cal
                ncal_ngt_neq[j, i, 1] = n_cal - idx_right
                ncal_ngt_neq[j, i, 2] = idx_right - idx_left

        return ncal_ngt_neq

    def predict_conf(self, x):
        p = self.predict(x, significance=None)
        label = p.argmax(axis=1)
        credibility = p.max(axis=1)
        for i, idx in enumerate(label):
            p[i, idx] = -np.inf
        confidence = 1 - p.max(axis=1)

        return np.array([label, confidence, credibility]).T


# -----------------------------------------------------------------------------
# Inductive conformal regressor
# -----------------------------------------------------------------------------
class IcpRegressor(BaseIcp, RegressorMixin):


    def __init__(self, nc_function, condition=None):
        super(IcpRegressor, self).__init__(nc_function, condition)

    def predict(self, x, significance=None):

        # TODO: interpolated p-values

        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], 2))

        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])
        y_predicted, error_predicted, index_score, nc_table = 0, 0, 0, 0
        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:  # MODIFY  the number of parameter get back
                p, y_predicted, error_predicted, index_score, nc_table = self.nc_function.predict(x[idx, :],
                                                                                        self.cal_scores[condition],
                                                                                        significance)
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction, y_predicted, error_predicted, index_score, nc_table


class OobCpClassifier(IcpClassifier):
    def __init__(self,
                 nc_function,
                 condition=None,
                 smoothing=True):
        super(OobCpClassifier, self).__init__(nc_function,
                                              condition,
                                              smoothing)

    def fit(self, x, y):
        super(OobCpClassifier, self).fit(x, y)
        super(OobCpClassifier, self).calibrate(x, y, False)

    def calibrate(self, x, y, increment=False):
        # Should throw exception (or really not be implemented for oob)
        pass


class OobCpRegressor(IcpRegressor):
    def __init__(self,
                 nc_function,
                 condition=None):
        super(OobCpRegressor, self).__init__(nc_function,
                                             condition)

    def fit(self, x, y):
        super(OobCpRegressor, self).fit(x, y)
        super(OobCpRegressor, self).calibrate(x, y, False)

    def calibrate(self, x, y, increment=False):
        # Should throw exception (or really not be implemented for oob)
        pass
