from random import sample, choice, shuffle

class PickStrategy:
	def Pick(self, swarm: dict[int, list[int]], n: int, exclude: list[int]) -> list[int]:
		"""
		Picks n peers from the swarm according to the chosen strategy.
		The caller should be removed from the list beforehand.
		"""
		return []

class LowStrategy(PickStrategy):
	"""Picks n peers with the lowest time values."""
	def Pick(self, swarm: dict[int, list[int]], n: int, exclude: list[int]) -> list[int]:
		candidates, picks = [], []
		ts = sorted(list(swarm.keys()))
		for t in ts:
			if len(candidates) >= 2 * n:
				break
			shuffle(swarm[t])
			candidates = candidates + swarm[t]
		for p in candidates:
			if p in exclude:
				continue
			picks.append(p)
			if len(picks) >= n:
				break
		return picks

class HighStrategy(PickStrategy):
	"""Picks n peers with the highest time values."""
	def Pick(self, swarm: dict[int, list[int]], n: int, exclude: list[int]) -> list[int]:
		candidates, picks = [], []
		ts = sorted(list(swarm.keys()))
		for t in reversed(ts):
			if len(candidates) >= 2 * n:
				break
			shuffle(swarm[t])
			candidates = candidates + swarm[t]
		for p in candidates:
			if p in exclude:
				continue
			picks.append(p)
			if len(picks) >= n:
				break
		return picks

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
