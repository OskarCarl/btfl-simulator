import csv

from . import peer, config, structs
import logging, sys

logger = logging.getLogger('sim.play')

class Executor:
	peers: list[peer.Peer]
	updates: list[list[structs.Update]] # Peer number -> Update number -> Gradients

	def __init__(self, peers: list[peer.Peer]):
		self.peers = peers
		self.updates = [[] for x in range(len(peers))]
		self.play: list[Step] = []
		logger.info("Set up Executor")

	def Execute(self):
		i, end = 0, len(self.play)
		for s in self.play:
			print("Step: {:05}/{}".format(i, end), end='\r', file=sys.stderr)
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
		return "Peer {} training".format(self.actor.num)

	def Exec(self, e: Executor):
		assert self.actor is not None
		upds = self.actor.Fit()
		for u in upds:
			e.updates[self.actor.num].append(u)

class ApplyStep(Step):
	source: int
	start: int = 0
	stop: int = config.EPOCHS
	
	def __init__(self, actor: peer.Peer | None, source: int, start: int, stop: int):
		assert actor is not None
		assert start is not None
		assert stop is not None
		self.actor = actor
		self.source = source
		if start != None:
			self.start = start
		if stop != None:
			self.stop = stop

	def __str__(self) -> str:
		assert self.actor is not None
		return "Peer {} applying update range [{}, {}) from peer {}".format(
			self.actor.num,
			self.start,
			self.stop,
			self.source
		)

	def Exec(self, e: Executor):
		assert self.actor is not None
		start = self.start if self.start >= 0 else self.actor.clock[self.source]
		stop = min(self.stop, len(e.updates[self.source]))
		for i in range(start, stop):
			self.actor.Apply(
				e.updates[self.source][i]
			)

class ClockStep(Step):
	def __init__(self):
		return

	def __str__(self) -> str:
		return "Printing peer clocks."

	def Exec(self, e: Executor):
		for p in e.peers:
			logger.info("Peer {} has clock {}".format(p.num, p.clock))
		return

class EvalStep(Step):
	def __str__(self) -> str:
		assert self.actor is not None
		return "Evaluating Peer {} with clock {}".format(self.actor.num, self.actor.clock)

	def Exec(self, e: Executor):
		assert self.actor is not None
		print("Peer {} - clock {} -".format(self.actor.num, self.actor.clock), end=' ')
		self.actor.Eval()

def Parse(file: str, e: Executor):
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
			elif row['action'] == 'apply':
				src = int(row['source'])
				start = int(row['start']) if (row['start'] != '' and row['start'] != None) else 0
				stop = int(row['stop']) if (row['stop'] != '' and row['stop'] != None) else 0
				s = ApplyStep(p, src, start, stop)
			elif row['action'] == 'eval':
				s = EvalStep(p)
			elif row['action'] == 'clocks':
				s = ClockStep()
			elif row['action'] == 'skip':
				continue
			else:
				logger.warn("Unrecognized action in line {}: {}".format(reader.line_num, row['action']))
				continue
			e.play.append(s)
