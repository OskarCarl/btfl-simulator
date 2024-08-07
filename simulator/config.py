import json
from dataclasses import dataclass, asdict

@dataclass
class PeerConfig:
	NUM_STEPS: int
	NUM_NEIGHBOURS: int
	NUM_ROTATE: int

	EPOCHS: int
	LEARNING_RATE: float
	SGD_MOMENTUM: float
	RETRAIN_FACTOR: float

	def __init__(self, numSteps: int, numNeighbours: int, numRotate: int, epochs: int, learningRate: float, sgdMomentum: float, retrainFactor: float):
		self.NUM_STEPS = numSteps
		self.NUM_NEIGHBOURS = numNeighbours
		self.NUM_ROTATE = numRotate
		self.EPOCHS = epochs
		self.LEARNING_RATE = learningRate
		self.SGD_MOMENTUM = sgdMomentum
		self.RETRAIN_FACTOR = retrainFactor

	def __str__(self) -> str:
		return json.dumps(asdict(self))

@dataclass
class SimulatorConfig:
	NUM_PEERS: int
	WEIGHTS_IDENTICAL: bool
	PEER_CONFIG: PeerConfig

	def __init__(self, numPeers: int, weigthsIdentical: bool, peerConfig: PeerConfig):
		self.NUM_PEERS = numPeers
		self.WEIGHTS_IDENTICAL = weigthsIdentical
		self.PEER_CONFIG = peerConfig

	def __str__(self) -> str:
		return json.dumps(asdict(self))
