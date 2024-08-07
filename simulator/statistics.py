import numpy as np
from math import sqrt
import numpy as np
from . import config
from peer import Peer
import logging

logger = logging.getLogger('simulator.statistics')

def Evaluate(peers: list[Peer]):
	for i in range(len(peers)):
		logger.info("Evaluating Peer {} with clock {}".format(peers[i].id, peers[i].time))
		peers[i].Eval()

def CollectWeights(peers: list[Peer]) -> list[list[np.ndarray]]:
	ws: list[list[np.ndarray]] = [[]] * len(peers)
	for p in peers:
		ws[p.id] = p.model.get_weights()
	return ws

def PerLayerAvgWeightDiff(ws: list[list[np.ndarray]]) -> np.ndarray:
	"""Calculates the relative average difference between pairs of model weights.
	"""
	logger.info("Calculating pair-wise (relative) avg difference in weights per layer")
	res = np.zeros(shape=(len(ws), len(ws), len(ws[0])))
	for a in range(len(ws)):
		for b in range(a, len(ws)):
			for layer in range(len(ws[0])):
				raw_diff = ws[a][layer] - ws[b][layer]
				diff = np.absolute(raw_diff) / (np.absolute(ws[a][layer]) + 1e-30) # TODO check this!
				mean = diff.flatten().mean()
				res[a, b, layer] = mean
	return res


def Dist(a: np.ndarray, b: np.ndarray) -> float:
	""" Computes the discrete Hellinger distance from a to b.
	    a and b are np arrays containing labels."""
	assert a.dtype == b.dtype

	a_labels, a_tmp = np.unique(a, return_counts=True)
	b_labels, b_tmp = np.unique(b, return_counts=True)

	labels = np.append(a_labels, b_labels)
	def fill(l: np.ndarray, c: np.ndarray) -> dict:
		ret = {i: 0 for i in labels}
		for n in range(len(l)):
			ret[l[n]] = c[n]
		return ret

	a_counts = fill(a_labels, a_tmp)
	b_counts = fill(b_labels, b_tmp)

	def p(x: a.dtype) -> float:
		return a_counts[x] / len(a)

	def q(x: a.dtype) -> float:
		return b_counts[x] / len(b)

	def square(i: float) -> float:
		return i * i

	def sum() -> float:
		s = 0.
		for x in labels:
			s += square(sqrt(p(x)) - sqrt(q(x)))
		return s

	return 1 / sqrt(2) * sqrt(sum())
