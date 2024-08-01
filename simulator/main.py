import tensorflow as tf
from keras import layers
from peer import config, peer
from . import play, dataset
from . import statistics as s
import logging

logger = logging.getLogger('sim')

class App:
	e: play.Executor

	def __init__(self, playFile, f=None):
		logger.info("TensorFlow version: {}".format(tf.__version__))
		# tf.debugging.experimental.enable_dump_debug_info('logs/', tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

		d = dataset.GetDataset(f)
		peers: list[peer.Peer] = []
		for i in range(config.NUM_PEERS):
			peers.append(peer.Peer(i, d[i], buildModel()))

		for i in range(config.NUM_PEERS - 1):
			for j in range(i+1, config.NUM_PEERS):
				d = s.Dist(peers[i].data.y_train, peers[j].data.y_train)
				logger.info("Distance between data sets {} and {}: {}".format(i, j, d))

		logger.info("Initializing weights to be idential")
		peers[0].model(peers[0].data.x_train[:2])
		w = peers[0].model.get_weights()
		for i in range(1, config.NUM_PEERS):
			peers[i].model.set_weights(w)

		# Set up the play
		self.e = play.Executor(peers)
		play.Parse(playFile, self.e)

	def run(self):
		self.e.Execute()
		# self.step2()
		s.Evaluate(self.e.peers)
		ws = s.CollectWeights(self.e.peers)
		avgDiff = s.PerLayerAvgWeightDiff(ws)
		print(avgDiff)
