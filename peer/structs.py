from numpy import ndarray, empty
from tensorflow import Tensor
import logging
from random import randrange

logger = logging.getLogger('simulator.data')

class Update:
	age: int
	weights: Tensor
	bias: Tensor | None # TODO: not implemented yet

	def __init__(self, age: int, weight: Tensor, bias: Tensor | None):
		self.age = age
		self.weights = weight
		self.bias = bias

class Data:
	x_train: ndarray
	y_train: ndarray
	x_test: ndarray
	y_test: ndarray
	stepSize: int
	stop: int

	def __init__(self, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray, steps: int):
		"""Initializes the data set. The size of each step is determined
		by the size of the overall set and the amount of steps.
		"""
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.stepSize = int(len(x_train) / steps)
		self.stop = self.stepSize

	def GetNext(self) -> tuple[ndarray, ndarray]:
		"""Retuns a growing number of data points and labels from the local data set.
		Consecutive calls walk through the whole data set, increasing by self.stepSize
		each time.
		If more steps are desired, their size can be adjusted indirectly
		via _steps_ during init."""
		logger.info("Using data set range [0, {})".format(self.stop))
		self.stop += self.stepSize
		return (
			self.x_train[:self.stop],
			self.y_train[:self.stop]
		)

	def GetRetrainSet(self, factor: float) -> tuple[ndarray, ndarray]:
		"""Returns all datapoints available up to now.
		"""
		num = int(factor * self.stop)
		start = randrange(0, self.stop - num)
		return (
			self.x_train[start:start + num],
			self.y_train[start:start + num]
		)
