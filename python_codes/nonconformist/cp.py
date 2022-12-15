from nonconformist.icp import *

# TODO: move contents from nonconformist.icp here

# -----------------------------------------------------------------------------
# TcpClassifier
# -----------------------------------------------------------------------------


class TcpClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, nc_function, condition=None, smoothing=True):
		self.train_x, self.train_y = None, None
		self.nc_function = nc_function
		super(TcpClassifier, self).__init__()

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

		self.smoothing = smoothing

		self.base_icp = IcpClassifier(
			self.nc_function,
			self.condition,
			self.smoothing
		)

		self.classes = None

	def fit(self, x, y):
		self.train_x, self.train_y = x, y
		self.classes = np.unique(y)

	def predict(self, x, significance=None):
		n_test = x.shape[0]
		n_train = self.train_x.shape[0]
		p = np.zeros((n_test, self.classes.size))
		for i in range(n_test):
			for j, y in enumerate(self.classes):
				train_x = np.vstack([self.train_x, x[i, :]])
				train_y = np.hstack([self.train_y, y])
				self.base_icp.fit(train_x, train_y)
				self.base_icp.calibrate(train_x, train_y)

				ncal_ngt_neq = self.base_icp._get_stats(x[i, :].reshape(1, x.shape[1]))

				ncal = ncal_ngt_neq[:, j, 0]
				ngt = ncal_ngt_neq[:, j, 1]
				neq = ncal_ngt_neq[:, j, 2]

				p[i, j] = calc_p(ncal - 1, ngt, neq - 1, self.smoothing)

		if significance is not None:
			return p > significance
		else:
			return p

	def predict_conf(self, x):
		p = self.predict(x, significance=None)
		label = p.argmax(axis=1)
		credibility = p.max(axis=1)
		for i, idx in enumerate(label):
			p[i, idx] = -np.inf
		confidence = 1 - p.max(axis=1)

		return np.array([label, confidence, credibility]).T
