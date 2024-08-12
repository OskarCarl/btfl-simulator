import peer, copy

class Swarm:
	peerlist: dict[int, peer.Peer]
	quicklist: dict[int, list[int]]
	timecounters: dict[int, int]

	def __init__(self) -> None:
		self.peerlist = {}
		self.quicklist = {}
		self.timecounters = {}

	def Add(self, p: peer.Peer) -> None:
		"""Adds a peer to the swarm list."""
		if p.id in self.peerlist:
			oldTime = self.timecounters[p.id]
			self.quicklist[oldTime].remove(p.id)
			if self.quicklist[oldTime] is None:
				del self.quicklist[oldTime]
		self.peerlist[p.id] = p
		self.timecounters[p.id] = p.age
		if not p.age in self.quicklist:
			self.quicklist[p.age] = []
		self.quicklist[p.age].append(p.id)

	def Get(self, id: int) -> peer.Peer:
		return self.peerlist[id]

	def GetQuickList(self) -> dict[int, list[int]]:
		"""Returns the quick look up list for the swarm."""
		return copy.deepcopy(self.quicklist)
