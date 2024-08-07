import logging

logger = logging.getLogger('tracker')

import peer
from .swarm import Swarm

class Tracker:
	swarm: Swarm

	def __init__(self):
		self.logger = logger
		self.s = Swarm()

	def Announce(self, p: peer.Peer) -> dict[int, list[int]]:
		self.s.Add(p)
		logger.info("Announced {}".format(p))
		return self.s.GetQuickList()

	def GetPeer(self, id: int) -> peer.Peer:
		return self.s.Get(id)
