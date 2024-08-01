import peer

class Swarm:
	l: dict[int, peer.Peer]
	q: dict[int, list[int]]

	def __init__(self) -> None:
		self.l = {}
		self.q = {}

	def Add(self, p: peer.Peer) -> None:
		"""Adds a peer to the swarm list.
		"""
		if p.id in self.l:
			oldTime = self.l[p.id].time
			self.q[oldTime].remove(p.id)
			if self.q[oldTime] is None:
				del self.q[oldTime]
		self.l[p.id] = p
		if not p.time in self.q:
			self.q[p.time] = []
		self.q[p.time].append(p.id)

	def Get(self, id: int) -> peer.Peer:
		return self.l[id]

	def GetQuickList(self, qTime: int, qId: int) -> dict[int, list[int]]:
		"""Returns the quick look up list for the swarm without the query source peer."""
		l = self.q
		l[qTime] = l[qTime].remove(qId)
		if l[qTime] is None:
			del l[qTime]
		return l
