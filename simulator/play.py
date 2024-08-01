import csv

from peer import peer, config, structs
import logging, sys

logger = logging.getLogger('sim.play')

class Executor:
	peers: list[peer.Peer]

	def __init__(self, peers: list[peer.Peer]):
		self.peers = peers
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

class ClockStep(Step):
	def __init__(self):
		return

	def __str__(self) -> str:
		return "Printing peer times."

	def Exec(self, e: Executor):
		for p in e.peers:
			logger.info("Peer {} has time {}".format(p.id, p.time))
		return

class EvalStep(Step):
	def __str__(self) -> str:
		assert self.actor is not None
		return "Evaluating Peer {} with time {}".format(self.actor.id, self.actor.time)

	def Exec(self, e: Executor):
		assert self.actor is not None
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
				s = CommunicateStep(p, src, start, stop)
			elif row['action'] == 'eval':
				s = EvalStep(p)
			elif row['action'] == 'times':
				s = ClockStep()
			elif row['action'] == 'skip':
				continue
			else:
				logger.warn("Unrecognized action in line {}: {}".format(reader.line_num, row['action']))
				continue
			e.play.append(s)
