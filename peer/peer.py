import numpy as np
import tensorflow as tf
import logging
from typing import Callable

from math import ceil
from . import config, structs
from .pick_strategy import *

class Peer:
	id: int
	data: structs.Data
	time: int
	tr = None
	swarm: dict[int, list[int]]
	neighbours: list
	picker: PickStrategy
	model: tf.keras.Model
	rng: dict[str, np.random.Generator]
	epoch: Callable[[np.ndarray, np.ndarray], None]
	logger: logging.Logger

	def __init__(self, n: int, tr, d: structs.Data, m: tf.keras.Model, l: logging.Logger):
		self.id = n
		self.time = 0
		self.data = d
		self.model = m
		self.rng = {
			'x': np.random.default_rng(seed=42),
			'y': np.random.default_rng(seed=42)
		}
		self.tr = tr
		self.neighbours = []
		self.logger = l
		self.picker = LowStrategy()

		def epoch(x: np.ndarray, y: np.ndarray):
			trainable_vars = self.model.trainable_variables
			x_shuf = self.rng['x'].permutation(x, axis=0)
			y_shuf = self.rng['y'].permutation(y, axis=0)
			@tf.function
			def train(x: np.ndarray, y: np.ndarray):
				with tf.GradientTape() as tape:
					y_pred = self.model(x, training=True)
					loss = self.model.compute_loss(y=y, y_pred=y_pred)

					# Compute gradients
					gradients = tape.gradient(loss, trainable_vars) # type: ignore
					self.model.optimizer.apply_gradients(zip(gradients, trainable_vars)) # type: ignore
			train(x_shuf, y_shuf)
		self.epoch = epoch

	def RotateNeighbours(self, n: int):
		assert n <= config.NUM_NEIGHBOURS
		assert self.tr is not None
		self.swarm = self.tr.Announce(self)
		if len(self.neighbours) == config.NUM_NEIGHBOURS:
			self.neighbours = self.neighbours[n:]
		new_neighbours = self.picker.Pick(self.swarm, n)
		for n in new_neighbours:
			self.neighbours.append(self.tr.GetPeer(n))
		self.logger.info("Peer {} rotated; new neighbours: {}".format(self.id, new_neighbours))

	def Communicate(self):
		u = structs.Update(
			self.time,
			self.model.get_weights(),
			tf.zeros_like(self.model.get_weights())
		)
		for n in self.neighbours:
			self.logger.info("Peer {} sending update to {}".format(self.id, n.id))
			n.OnReceiveModel(u)

	def OnReceiveModel(self, u: structs.Update):
		assert self.tr is not None
		self.swarm = self.tr.Announce(self)
		# TODO: actually apply the update
		#self.model.
		if config.RETRAIN_FACTOR > 0.0:
			values, labels = self.data.GetRetrainSet()
			self.epoch(values, labels)

		self.time = max(self.time, u.time)

	def Fit(self, epochs=7):
		values, labels = self.data.GetNext()
		for _ in range(epochs):
			self.epoch(values, labels)
			self.time += 1

	def Eval(self):
		"""Runs the evaluate() fn on the local model, logging its output."""
		metrics = self.model.evaluate(
			self.data.x_test,
			self.data.y_test,
			verbose=0,
			return_dict=True
		)
		self.logger.info("peer {}; time {}; metrics: {}".format(self.id, self.time, metrics))

	# def getCallback(self) -> tf.keras.callbacks.TensorBoard:
	# 	logpath = "logs/fit/{}-peer{}".format(
	# 		datetime.now().strftime("%Y%m%d-%H%M%S"),
	# 		self.num
	# 		)
	# 	return tf.keras.callbacks.TensorBoard(log_dir=logpath)
