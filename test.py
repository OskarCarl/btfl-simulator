from peer import Peer
from tracker import Tracker
from simulator import GetDataset, BuildModel
import logging

logging.basicConfig(filename="test.log", level=logging.INFO)

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
	t.Announce(p)

for p in ps:
	p.Fit()
	p.Communicate()

for p in ps:
	p.RotateNeighbours(2)
	p.Fit()
	p.Communicate()

for p in ps:
	p.Eval()
