from peer import Peer
from tracker import Tracker
from simulator import GetDataset, BuildModel
import logging

# TODO fix logging!
logging.getLogger()

tl = logging.getLogger("tracker")
tl.setLevel(logging.INFO)
t = Tracker(tl)

data = GetDataset()

pl = logging.getLogger("peers")
pl.setLevel(logging.INFO)
ps = []
for i in range(5):
	p = Peer(i, t, data[i], BuildModel(), pl)
	ps.append(p)

for p in ps:
	p.RotateNeighbours(2)
