from random import sample, choice, shuffle

class PickStrategy:
	def Pick(self, swarm: dict[int, list[int]], n: int) -> list[int]:
		return []

class LowStrategy(PickStrategy):
	"""Picks n peers with the lowest time values."""
	def Pick(self, swarm: dict[int, list[int]], n: int) -> list[int]:
		picks = []
		ts = sorted(list(swarm.keys()))
		for t in reversed(ts):
			if len(picks) >= n:
				break
			shuffle(swarm[t])
			picks = picks + swarm[t]
			if t == ts[0]:
				break
		return picks[:min(len(picks), n)]

class HighStrategy(PickStrategy):
	"""Picks n peers with the highest time values."""
	def Pick(self, swarm: dict[int, list[int]], n: int) -> list[int]:
		picks = []
		ts = sorted(list(swarm.keys()))
		for t in ts:
			if len(picks) >= n:
				break
			shuffle(swarm[t])
			picks = picks + swarm[t]
			if t == ts[-1]:
				break
		return picks[:min(len(picks), n)]

# TODO implement?
# class RandomStrategy(PickStrategy):
# 	def Pick(self, swarm: dict[int, list[int]], n: int) -> list[int]:
# 		picks = []
# 		ts = sample(list(swarm.keys()), min(n, len(swarm)))
# 		available = 0
# 		for t in ts:
# 			picks.append(choice(swarm[t]))
# 			available += len(swarm[t])
# 		while len(picks) < n:

# 		return picks
