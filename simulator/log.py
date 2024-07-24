import logging

loggers = [
	'sim', 'sim.data', 'sim.statistics', 'sim.peer', 'sim.play',
	'peer'
]

class LogContextManager:
	filename: str
	logger: logging.Logger
	handler: logging.FileHandler

	def __init__(self, filename):
		self.filename = filename
		self.logger = logging.getLogger('sim')

	def __enter__(self):
		self.handler = logging.FileHandler(self.filename, 'w')
		self.logger.addHandler(self.handler)
		self.logger.setLevel(logging.INFO)

	def __exit__(self, exc_type, exc_value, exc_traceback):
		self.logger.removeHandler(self.handler)
		self.handler.flush()
		self.handler.close()
