import argparse, glob, logging, os
from datetime import datetime

logging.basicConfig(level=logging.INFO)

from simulator import config
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

parser = argparse.ArgumentParser(
	description='Small simulator to train an distribute ML models in a BitTorrent network.'
)

parser.add_argument(
	'-p', '--play',
	help="Path to the play file(s) for this simulation. Globs are supported.",
	required=True,
	nargs="+"
)

parser.add_argument(
	'--numpeers',
	help="The number of peers to create. (5)",
	type=int,
	default=5
)

parser.add_argument(
	'--numneighbours',
	help="The number of neighbours per peer. (2)",
	type=int,
	default=2
)

parser.add_argument(
	'--numrotate',
	help="The number of neighbours a peer rotates. (2)",
	type=int,
	default=2
)

parser.add_argument(
	'--identicalweights',
	help="If set, all peers will have the same weights. (False)",
	action='store_true',
	default=False
)

parser.add_argument(
	'--steps',
	help="Number of steps that can be taken per peer. Since the amount of data is fixed more stepes means smaller ones. (10)",
	type=int,
	default=10
)

parser.add_argument(
	'--epochs',
	help="The number of epochs per step. (7)",
	type=int,
	default=7,
)

parser.add_argument(
	'--retrainfactor',
	help="Ratio of known data to use for retraining after each update apply. (0.2)",
	type=float,
	default=0.2
)

parser.add_argument(
	'--learnrate',
	help="Learning rate to use. (0.01)",
	type=float,
	default=0.01
)

parser.add_argument(
	'--datafile',
	help='Data file to use. (none)',
	type=str,
	default=None
)

parser.add_argument(
	'-l', '--logdir',
	help="Directory where logs should be stored. If omitted, logs are printed.",
	default=None
)

args = parser.parse_args()

c = config.SimulatorConfig(
	args.numpeers,
	args.identicalweights,
	config.PeerConfig(
		args.steps,
		args.numneighbours,
		args.numrotate,
		args.epochs,
		args.learnrate,
		0.0,
		args.retrainfactor
	)
)

plays: list[str] = []
for p in args.play:
	plays.extend(glob.iglob(p))
plays.sort()
print("Running plays {}".format(plays))
if args.datafile is not None:
	print("Using data file {}".format(args.datafile))
print("Running with config {}".format(c))

from os import path
from simulator import play

def normalize(p: str) -> str:
	if p is None:
		return None
	p = path.basename(p)
	return path.splitext(p)[0]

for p in plays:
	print("Run {}.".format(p))
	base = ""
	if args.logdir is not None:
		if not path.exists(args.logdir):
			os.makedirs(args.logdir)
		ts = datetime.now().strftime("%Y%m%d-%H%M%S")
		base = path.normpath("{}/{}_{}".format(args.logdir, ts, normalize(p)))
		if path.exists("{}.done".format(base)):
			print("Skipping {} as it is marked done.".format(base))
			continue
		print("Saving logs to {}.log".format(base))
		logging.basicConfig(filename="{}.log".format(base), level=logging.INFO, force=True)

	e = play.Setup(c, p, args.datafile)
	e.Execute()

	if args.logdir is not None:
		open("{}.done".format(base), 'x').close()
	print("Run done.")
