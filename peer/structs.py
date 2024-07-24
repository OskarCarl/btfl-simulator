from numpy import ndarray, empty
from tensorflow import Tensor
import logging
from . import config
from random import randrange

logger = logging.getLogger('sim.data')

class Update:
	time: int
	weights: Tensor
	bias: Tensor

	def __init__(self, t: int, g: Tensor, s: Tensor, c: int):
		self.time = t
		self.weights = g
		self.bias = s

class Data:
	x_train: ndarray
	y_train: ndarray
	x_test: ndarray
	y_test: ndarray
	stepSize: int
	stepNum: int

	def __init__(self, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray, steps: int):
		"""Initializes the data set. The size of each step is determined
		by the size of the overall set and the amount of steps.
		"""
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.stepSize = int(len(x_train) / steps)
		self.stepNum = 0

	def GetNext(self) -> tuple[ndarray, ndarray]:
		"""Retuns the next (up to) 3 * self.stepSize data points and labels
		from the local data set. Consecutive calls walk through the whole
		set in a sliding window.
		Returns [] and logs a warning once it reaches the end.
		If more steps are desired, their size can be adjusted indirectly
		via _steps_ during init."""
		(start, stop) = self.getStartStop()
		if stop > len(self.x_train):
			logger.warn("Data set is exhausted!")
			return (empty(0), empty(0))
		else:
			logger.info("Using data set range [{}, {})".format(start, stop))
		self.stepNum += 1
		return (
			self.x_train[start:stop],
			self.y_train[start:stop]
		)

	def getStartStop(self) -> tuple[int, int]:
		start = max(self.stepNum - 2, 0) * self.stepSize
		stop = (self.stepNum + 1) * self.stepSize - 1
		return (start, stop)

	def GetRetrainSet(self) -> tuple[ndarray, ndarray]:
		"""Returns all datapoints available up to now.
		"""
		(_, stop) = self.getStartStop()
		num = int(config.RETRAIN_FACTOR * stop)
		start = randrange(0, stop - num)
		return (
			self.x_train[start:start + num],
			self.y_train[start:start + num]
		)
