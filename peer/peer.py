import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger('peer')
from typing import Callable

from . import structs
from .pick_strategy import *
from simulator import config

class Peer:
	id: int
	conf: config.PeerConfig
	data: structs.Data
	time: int
	tr = None
	swarm: dict[int, list[int]]
	neighbours: list
	picker: PickStrategy
	model: tf.keras.Model
	rng: dict[str, np.random.Generator]
	epoch: Callable[[np.ndarray, np.ndarray], None]

	def __init__(self, n: int, tr, d: structs.Data, m: tf.keras.Model, c: config.PeerConfig):
		self.id = n
		self.time = 0
		self.conf = c
		self.data = d
		self.model = m
		self.rng = {
			'x': np.random.default_rng(seed=42),
			'y': np.random.default_rng(seed=42)
		}
		self.tr = tr
		self.neighbours = []
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

	def __str__(self) -> str:
		return "Peer {} with time {}".format(self.id, self.time)

	def RotateNeighbours(self, n: int):
		assert n <= self.conf.NUM_NEIGHBOURS
		assert self.tr is not None
		self.swarm = self.tr.Announce(self)
		self.swarm[self.time].remove(self.id)
		if len(self.neighbours) == self.conf.NUM_NEIGHBOURS:
			self.neighbours = self.neighbours[n:]
		new_neighbours = self.picker.Pick(self.swarm, n)
		for n in new_neighbours:
			self.neighbours.append(self.tr.GetPeer(n))
		logger.info("Peer {} rotated; new neighbours {}".format(self.id, new_neighbours))

	def Communicate(self):
		u = structs.Update(
			self.time,
			self.model.get_weights(),
			tf.zeros(shape=(1,1)) # TODO: placeholder
		)
		for n in self.neighbours:
			logger.info("Peer {} sending update to {}".format(self.id, n.id))
			n.OnReceiveModel(u)

	def OnReceiveModel(self, u: structs.Update):
		"""
		Implementation following Alg. 1+2 of Hegedűs 2020 (https://doi.org/10.1016/j.jpdc.2020.10.006).
		Currently ignores biases.
		"""
		assert self.tr is not None

		# Merge
		w = self.model.get_weights()
		for i in range(len(w)):
			w[i] = (w[i] * self.time + u.weights[i] * u.time) / (self.time + u.time)
		self.model.set_weights(w)
		self.time = max(self.time, u.time)

		# Update
		if self.conf.RETRAIN_FACTOR > 0.0:
			values, labels = self.data.GetRetrainSet(self.conf.RETRAIN_FACTOR)
			self.epoch(values, labels)

		self.swarm = self.tr.Announce(self)

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
		logger.info("peer {}; time {}; metrics {}".format(self.id, self.time, metrics))

	# def getCallback(self) -> tf.keras.callbacks.TensorBoard:
	# 	logpath = "logs/fit/{}-peer{}".format(
	# 		datetime.now().strftime("%Y%m%d-%H%M%S"),
	# 		self.num
	# 		)
	# 	return tf.keras.callbacks.TensorBoard(log_dir=logpath)
