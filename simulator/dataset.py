from . import config, structs
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger('sim.data')

def Get(storedfile=None) -> list[structs.Data]:
	d: list[structs.Data] = []
	if storedfile is not None:
		loaded = np.load(storedfile)
		x_test = loaded['x_test'] / 255.0
		for peer in range(config.NUM_PEERS):
			x_train = loaded['x_{}'.format(peer)] / 255.0
			d.append(structs.Data(
				x_train, loaded['y_{}'.format(peer)],
				x_test, loaded['y_test'],
				config.NUM_STEPS))
		logger.info("Loaded data set {}.".format(storedfile))
		print(loaded['dist'])
	else:
		# Distribute the dataset evenly among the peers
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
		x_train, x_test = x_train / 255.0, x_test / 255.0

		for i in range(config.NUM_PEERS):
			start = int(len(x_train) / config.NUM_PEERS) * i
			stop = int(len(x_train) / config.NUM_PEERS) * (i + 1) - 1 # We lose some data at the end; doesn't matter
			logger.info("Set up data object of mixed labels for peer {} with {} items.".format(i, stop-start))
			d.append(structs.Data(x_train[start:stop], y_train[start:stop], x_test, y_test, config.NUM_STEPS))

	return d
