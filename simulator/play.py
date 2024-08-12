import csv

from peer import peer
import logging, sys

from . import dataset, config, model, statistics
import tracker

logger = logging.getLogger('simulator.play')

class Executor:
	peers: list[peer.Peer]
	log: logging.Logger

	def __init__(self, peers: list[peer.Peer]):
		self.peers = peers
		self.play: list[Step] = []

	def Execute(self):
		i, end = 0, len(self.play)
		for s in self.play:
			print("Step: {:06}/{}".format(i+1, end), end='\r', file=sys.stderr)
			logger.info(s)
			s.Exec(self)
			i += 1
		print('', file=sys.stderr)

class Step:
	actor: peer.Peer | None

	def __init__(self, actor: peer.Peer | None):
		self.actor = actor

	def Exec(self, e: Executor):
		return

class FitStep(Step):
	def __str__(self) -> str:
		assert self.actor is not None
		return "Peer {} training".format(self.actor.id)

	def Exec(self, e: Executor):
		assert self.actor is not None
		self.actor.Fit()

class CommunicateStep(Step):
	def __str__(self) -> str:
		assert self.actor is not None
		return "Peer {} sending updates to neighbours".format(
			self.actor.id,
		)

	def Exec(self, e: Executor):
		assert self.actor is not None
		self.actor.Communicate()

class RotateStep(Step):
	n: int

	def __init__(self, actor: peer.Peer, n: int | None):
		self.actor = actor
		if n is None:
			self.n = actor.conf.NUM_ROTATE
		else:
			self.n = n

	def __str__(self) -> str:
		assert self.actor is not None
		return "Peer {} rotating neighbours".format(
			self.actor.id,
		)

	def Exec(self, e: Executor):
		assert self.actor is not None
		self.actor.RotateNeighbours(self.n)

class EvalStep(Step):
	def __str__(self) -> str:
		assert self.actor is not None
		return "Evaluating Peer {}".format(self.actor.id)

	def Exec(self, e: Executor):
		assert self.actor is not None
		self.actor.Eval()

def parse(file: str, e: Executor):
	with open(file=file) as f:
		reader = csv.DictReader(f)
		for row in reader:
			if row['peer'] != '':
				if int(row['peer']) >= len(e.peers):
					logger.warn("Unrecognized peer in line {}: {}".format(reader.line_num, row['peer']))
					continue
				p = e.peers[int(row['peer'])]
			else:
				p = None
			s = None
			if row['action'] == 'fit':
				s = FitStep(p)
			elif row['action'] == 'communicate':
				s = CommunicateStep(p)
			elif row['action'] == 'rotate':
				assert p is not None
				n = int(row['num']) if (row['num'] != '' and row['num'] != None) else None
				s = RotateStep(p, n)
			elif row['action'] == 'eval':
				s = EvalStep(p)
			elif row['action'] == 'skip':
				continue
			else:
				logger.warn("Unrecognized action in line {}: {}".format(reader.line_num, row['action']))
				continue
			e.play.append(s)

def Setup(conf: config.SimulatorConfig, pf: str, df: str | None) -> Executor:
	logger.info("Set up Executor with config {}".format(conf))
	d = dataset.GetDataset(conf.NUM_PEERS, conf.PEER_CONFIG.NUM_STEPS, df)
	tr = tracker.Tracker()
	peers: list[peer.Peer] = []
	for i in range(conf.NUM_PEERS):
		p = peer.Peer(
				i, tr, d[i], model.BuildModel(
					conf.PEER_CONFIG.LEARNING_RATE, conf.PEER_CONFIG.SGD_MOMENTUM
				), conf.PEER_CONFIG
			)
		peers.append(p)
		tr.Announce(p)

	for i in range(conf.NUM_PEERS - 1):
		for j in range(i+1, conf.NUM_PEERS):
			d = statistics.Dist(peers[i].data.y_train, peers[j].data.y_train)
			logger.info("Distance between data sets {} and {}: {}".format(i, j, d))

	if conf.WEIGHTS_IDENTICAL:
		logger.info("Initializing weights to be identical")
		peers[0].model(peers[0].data.x_train[:2])
		w = peers[0].model.get_weights()
		for i in range(1, conf.NUM_PEERS):
			peers[i].model.set_weights(w)

	# Set up the play
	e = Executor(peers)
	parse(pf, e)

	return e
