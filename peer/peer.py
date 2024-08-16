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
	age: int
	tr = None
	swarm: dict[int, list[int]]
	neighbours: list
	picker: PickStrategy
	model: tf.keras.Model
	rng: dict[str, np.random.Generator]
	epoch: Callable[[np.ndarray, np.ndarray], None]

	def __init__(self, n: int, tr, d: structs.Data, m: tf.keras.Model, c: config.PeerConfig):
		self.id = n
		self.age = 0
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
		return '{{"peer": {}, "age": {}, "neighbours": {}}}'.format(self.id, self.age, [n.id for n in self.neighbours])

	def RotateNeighbours(self, n: int):
		assert n <= self.conf.NUM_NEIGHBOURS
		assert self.tr is not None
		self.swarm = self.tr.Announce(self)
		self.swarm[self.age].remove(self.id)
		if len(self.neighbours) > (self.conf.NUM_NEIGHBOURS - n):
			self.neighbours = self.neighbours[-(self.conf.NUM_NEIGHBOURS - n):]
		new_neighbours = self.picker.Pick(self.swarm, n, [n.id for n in self.neighbours])
		for n in new_neighbours:
			self.neighbours.append(self.tr.GetPeer(n))
		logger.info('{{"peer": {}, "action": "rotate", "age": {}, "neighbours": {}, "new": {}}}'.format(self.id, self.age, [n.id for n in self.neighbours], new_neighbours))

	def Communicate(self):
		u = structs.Update(
			self.age,
			self.model.get_weights(),
			None
		)
		logger.info('{{"peer": {}, "action": "communicate", "age": {}, "sending_to": {}}}'.format(self.id, self.age, [n.id for n in self.neighbours]))
		for n in self.neighbours:
			n.OnReceiveModel(u)

	def OnReceiveModel(self, u: structs.Update):
		"""
		Implementation following Alg. 1+2 of Hegedűs 2020 (https://doi.org/10.1016/j.jpdc.2020.10.006).
		Biases are included in the weights tensor list.
		"""
		assert self.tr is not None

		# Merge
		w = self.model.get_weights()
		for i in range(len(w)):
			w[i] = (w[i] * self.age + u.weights[i] * u.age) / max(1, self.age + u.age)
		self.model.set_weights(w)
		self.age = max(self.age, u.age) + 1

		# Update
		if self.conf.RETRAIN_FACTOR > 0.0:
			values, labels = self.data.GetRetrainSet(self.conf.RETRAIN_FACTOR)
			for _ in range(3):
				self.epoch(values, labels)

	def Fit(self, epochs=7):
		values, labels = self.data.GetNext()
		for _ in range(epochs):
			self.epoch(values, labels)
		self.age += 1

	def Eval(self):
		"""Runs the evaluate() fn on the local model, logging its output."""
		metrics = self.model.evaluate(
			self.data.x_test,
			self.data.y_test,
			verbose=0,
			return_dict=True
		)
		logger.info('{{"peer": {}, "action": "eval", "age": {}, "metrics": {}}}'.format(self.id, self.age, metrics))

	# def getCallback(self) -> tf.keras.callbacks.TensorBoard:
	# 	logpath = "logs/fit/{}-peer{}".format(
	# 		datetime.now().strftime("%Y%m%d-%H%M%S"),
	# 		self.num
	# 		)
	# 	return tf.keras.callbacks.TensorBoard(log_dir=logpath)
