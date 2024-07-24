from logging import Logger
from ..peer import peer

class Tracker:
	swarm: dict[int, peer.Peer]
	swarm_quick: dict[int, int]
	logger: Logger

	def __init__(self, logger: Logger):
		self.logger = logger
		self.swarm = {}
		self.swarm_quick = {}

	def Announce(self, peer: peer.Peer) -> dict[int, int]:
		self.swarm[peer.id] = peer
		self.swarm_quick[peer.id] = peer.time
		self.logger.info("Announced {}".format(peer))
		return self.swarm_quick

	def GetPeer(self, id: int) -> peer.Peer:
		return self.swarm[id]
