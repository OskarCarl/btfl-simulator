import argparse, glob
from app import config

parser = argparse.ArgumentParser(
	description='Small simulator to train an distribute ML models in a bittorrent network.'
)

parser.add_argument(
	'-n', '--num',
	help="The number of peers to create. ({})".format(config.NUM_PEERS),
	type=int,
	default=config.NUM_PEERS
)

parser.add_argument(
	'-s', '--steps',
	help="Number of steps that can be taken per peer. Since the amount of data is fixed more stepes means smaller ones. ({})".format(config.NUM_STEPS),
	type=int,
	default=config.NUM_STEPS
)

parser.add_argument(
	'-e', '--epochs',
	help="The number(s) of epochs per step. ({})".format([config.EPOCHS]),
	default=[config.EPOCHS],
	nargs='+'
)

parser.add_argument(
	'-b', '--batchsize',
	help="Size of the batches in an epoch. ({})".format(config.BATCHSIZE),
	type=int,
	default=config.BATCHSIZE
)

parser.add_argument(
	'-r', '--retrain',
	help="Ratio of known data to use for retraining after each update apply. ({})".format(config.RETRAIN_FACTOR),
	type=float,
	default=config.RETRAIN_FACTOR
)

parser.add_argument(
	'-a', '--learnrate',
	help="Learning rate to use. ({})".format(config.LEARNING_RATE),
	type=float,
	default=config.LEARNING_RATE
)

parser.add_argument(
	'-f', '--file',
	help='Data file(s) to use. (none)',
	default=[None],
	nargs='+'
)

parser.add_argument(
	'-l', '--logdir',
	help="Directory where logs should be stored. If omitted, logs are printed.",
	default=None
)

args = parser.parse_args()

config.NUM_PEERS = args.num
config.NUM_STEPS = args.steps
config.BATCHSIZE = args.batchsize
config.LEARNING_RATE = args.learnrate
config.RETRAIN_FACTOR = args.retrain

plays: list[str] = []
for p in args.play:
	plays.extend(glob.iglob(p))
plays.sort()
print("Running plays {}".format(plays))

datafiles: list[str] = []
for f in args.file:
	if f is None:
		datafiles.append(None)
	else:
		datafiles.extend(glob.iglob(f))
datafiles.sort()
print("Using data files {}".format(datafiles))

epochs: list[int] = []
for e in args.epochs:
	epochs.append(int(e))
print("Running with {} epochs.".format(epochs))

print("Running with: {} peers, {} steps, {} batchsize, {} learningrate.".format(
	config.NUM_PEERS, config.NUM_STEPS, config.BATCHSIZE, config.LEARNING_RATE))

from app import main, log
from os import path
from contextlib import redirect_stdout

def normalize(p: str) -> str:
	if p is None:
		return None
	p = path.basename(p)
	return path.splitext(p)[0]

for p in plays:
	for f in datafiles:
		for e in epochs:
			print("Run {} with {} and {} epochs.".format(p, f, e))
			config.EPOCHS = e
			if args.logdir is not None:
				base = path.normpath("{}/{}_{}_e{}".format(args.logdir, normalize(p), normalize(f), e))
				if path.exists("{}.done".format(base)):
					print("Skipping {} as it is marked done.".format(base))
					continue
				out = "{}.out.log".format(base)
				info = "{}.info.log".format(base)
				print("Saving logs to {}.{{out|info}}.log".format(base))
				with open(out, 'w') as o:
					with redirect_stdout(o), log.LogContextManager(info):
						a = main.App(p, f=f)
						a.run()
				open("{}.done".format(base), 'x').close()
			else:
				a = main.App(p, f=f)
				a.run()
			print("Run done.")
