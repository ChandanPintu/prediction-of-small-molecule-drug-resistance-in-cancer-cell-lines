#!/usr/bin/env python

"""
Nonconformity functions.
"""

# Authors: Henrik Linusson

from __future__ import division

import abc
import numpy as np
import sklearn.base
from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.base import OobClassifierAdapter, OobRegressorAdapter


# -----------------------------------------------------------------------------
# Error functions
# -----------------------------------------------------------------------------


class ClassificationErrFunc(object):


    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(ClassificationErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):
        pass


class RegressionErrFunc(object):


    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(RegressionErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):  # , norm=None, beta=0):

        pass

    @abc.abstractmethod
    def apply_inverse(self, nc, significance):  # , norm=None, beta=0):

        pass


class InverseProbabilityErrFunc(ClassificationErrFunc):

    def __init__(self):
        super(InverseProbabilityErrFunc, self).__init__()

    def apply(self, prediction, y):
        prob = np.zeros(y.size, dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= prediction.shape[1]:
                prob[i] = 0
            else:
                prob[i] = prediction[i, int(y_)]
        return 1 - prob


class MarginErrFunc(ClassificationErrFunc):

    def __init__(self):
        super(MarginErrFunc, self).__init__()

    def apply(self, prediction, y):
        prob = np.zeros(y.size, dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= prediction.shape[1]:
                prob[i] = 0
            else:
                prob[i] = prediction[i, int(y_)]
                prediction[i, int(y_)] = -np.inf
        return 0.5 - ((prob - prediction.max(axis=1)) / 2)


class AbsErrorErrFunc(RegressionErrFunc):

    def __init__(self):
        super(AbsErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return np.abs(prediction - y)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        # TODO: should probably warn against too few calibration examples
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])


class SignErrorErrFunc(RegressionErrFunc):

    def __init__(self):
        super(SignErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return (prediction - y)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        upper = int(np.floor((significance / 2) * (nc.size + 1)))
        lower = int(np.floor((1 - significance / 2) * (nc.size + 1)))
        # TODO: should probably warn against too few calibration examples
        upper = min(max(upper, 0), nc.size - 1)
        lower = max(min(lower, nc.size - 1), 0)
        return np.vstack([-nc[lower], nc[upper]])


# -----------------------------------------------------------------------------
# Base nonconformity scorer
# -----------------------------------------------------------------------------
class BaseScorer(sklearn.base.BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseScorer, self).__init__()

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def score(self, x, y=None):
        pass


class RegressorNormalizer(BaseScorer):
    def __init__(self, base_model, normalizer_model, err_func):
        super(RegressorNormalizer, self).__init__()
        self.base_model = base_model
        self.normalizer_model = normalizer_model
        self.err_func = err_func

    def fit(self, x, y):
        residual_prediction = self.base_model.predict(x)
        residual_error = np.abs(self.err_func.apply(residual_prediction, y))
        residual_error += 0.00001  # Add small term to avoid log(0)
        log_err = np.log(residual_error)
        self.normalizer_model.fit(x, log_err)

    def score(self, x, y=None):
        norm = np.exp(self.normalizer_model.predict(x))
        return norm


class NcFactory(object):
    @staticmethod
    def create_nc(model, err_func=None, normalizer_model=None, oob=False,
                  fit_params=None, fit_params_normalizer=None):
        if normalizer_model is not None:
            normalizer_adapter = RegressorAdapter(normalizer_model, fit_params_normalizer)
        else:
            normalizer_adapter = None

        if isinstance(model, sklearn.base.ClassifierMixin):
            err_func = MarginErrFunc() if err_func is None else err_func
            if oob:
                c = sklearn.base.clone(model)
                c.fit([[0], [1]], [0, 1])
                if hasattr(c, 'oob_decision_function_'):
                    adapter = OobClassifierAdapter(model, fit_params)
                else:
                    raise AttributeError('Cannot use out-of-bag '
                                         'calibration with {}'.format(
                        model.__class__.__name__
                    ))
            else:
                adapter = ClassifierAdapter(model, fit_params)

            if normalizer_adapter is not None:
                normalizer = RegressorNormalizer(adapter,
                                                 normalizer_adapter,
                                                 err_func)
                return ClassifierNc(adapter, err_func, normalizer)
            else:
                return ClassifierNc(adapter, err_func)

        elif isinstance(model, sklearn.base.RegressorMixin):
            err_func = AbsErrorErrFunc() if err_func is None else err_func
            if oob:
                c = sklearn.base.clone(model)
                c.fit([[0], [1]], [0, 1])
                if hasattr(c, 'oob_prediction_'):
                    adapter = OobRegressorAdapter(model, fit_params)
                else:
                    raise AttributeError('Cannot use out-of-bag '
                                         'calibration with {}'.format(model.__class__.__name__))
            else:
                adapter = RegressorAdapter(model, fit_params)

            if normalizer_adapter is not None:
                normalizer = RegressorNormalizer(adapter,
                                                 normalizer_adapter,
                                                 err_func)
                return RegressorNc(adapter, err_func, normalizer)
            else:
                return RegressorNc(adapter, err_func)


class BaseModelNc(BaseScorer):

    def __init__(self, model, err_func, normalizer=None, beta=0):
        super(BaseModelNc, self).__init__()
        self.err_func = err_func
        self.model = model
        self.normalizer = normalizer
        self.beta = beta

        # If we use sklearn.base.clone (e.g., during cross-validation),
        # object references get jumbled, so we need to make sure that the
        # normalizer has a reference to the proper model adapter, if applicable.
        if (self.normalizer is not None and
                hasattr(self.normalizer, 'base_model')):
            self.normalizer.base_model = self.model

        self.last_x, self.last_y = None, None
        self.last_prediction = None
        self.clean = False

    def fit(self, x, y):

        self.model.fit(x, y)
        if self.normalizer is not None:
            self.normalizer.fit(x, y)
        self.clean = False

    def score(self, x, y=None):

        prediction = self.model.predict(x)
        n_test = x.shape[0]
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)
        return self.err_func.apply(prediction, y) / norm


# -----------------------------------------------------------------------------
# Classification nonconformity scorers
# -----------------------------------------------------------------------------
class ClassifierNc(BaseModelNc):

    def __init__(self,
                 model,
                 err_func=MarginErrFunc(),
                 normalizer=None,
                 beta=0):
        super(ClassifierNc, self).__init__(model,
                                           err_func,
                                           normalizer,
                                           beta)


# -----------------------------------------------------------------------------
# Regression nonconformity scorers
# -----------------------------------------------------------------------------
class RegressorNc(BaseModelNc):

    def __init__(self,
                 model,
                 err_func=AbsErrorErrFunc(),
                 normalizer=None,
                 beta=0):
        super(RegressorNc, self).__init__(model,
                                          err_func,
                                          normalizer,
                                          beta)

    def predict(self, x, nc, significance=None):

        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)


        if significance:
            intervals = np.zeros((x.shape[0], 2))
            err_dist = self.err_func.apply_inverse(nc, significance)
            index_score = err_dist[0][0]
            err_dist = np.hstack([err_dist] * n_test)
            err_dist *= norm

            intervals[:, 0] = prediction - err_dist[0, :]
            intervals[:, 1] = prediction + err_dist[1, :]
            # TODO modify here to return norm (error model) and  error_dist (alpha_s, before multiply by norm)
            # in this way I will get values for error model and check their behaviour
            return intervals, prediction, norm, index_score, nc
        else:
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                index_score = err_dist[0][0]
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals, prediction, norm, index_score, nc
