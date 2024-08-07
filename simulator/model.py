import tensorflow as tf
from keras import layers

def BuildModel(lr: float, momentum: float) -> tf.keras.Model:
	inputs = tf.keras.Input(shape=(28,28))
	flatten = layers.Flatten()(inputs)
	hidden = layers.Dense(128, activation='relu')(flatten)
	outputs = layers.Dense(10)(hidden)

	model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
	model.compile(
		optimizer=tf.keras.optimizers.SGD(
			learning_rate=lr,
			momentum=momentum
		),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])

	return model
