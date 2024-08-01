from logging import Logger

import peer
from .swarm import Swarm

class Tracker:
	swarm: Swarm
	logger: Logger

	def __init__(self, logger: Logger):
		self.logger = logger
		self.s = Swarm()

	def Announce(self, p: peer.Peer) -> dict[int, list[int]]:
		self.s.Add(p)
		self.logger.info("Announced {}".format(p))
		return self.s.GetQuickList(p.time, p.id)

	def GetPeer(self, id: int) -> peer.Peer:
		return self.s.Get(id)
