"""
Generated by Claude 3.5 Sonnet, adapted by Oskar Carl
"""

import json, sys
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np

matplotlib.use('webagg')

NUM_PEERS = 30

def extract_data(file):
	# Accuracy, age, line number
	all = [[], [], [], []]
	# Final accuracy, age
	final = [[0 for _ in range(NUM_PEERS)], [0. for _ in range(NUM_PEERS)]]
	diversity = [[False for _ in range(NUM_PEERS)] for _ in range(NUM_PEERS)]
	with open(file, 'r') as file:
		for i, line in enumerate(file):
			# Split the line and extract the JSON part
			parts = line.split(':', 2)
			if len(parts) == 3:
				try:
					# Parse the JSON data
					data = json.loads(parts[2].strip().replace("'", '"'))

					if data['action'] == 'eval':
						all[0].append(data['metrics']['accuracy'])
						all[1].append(data['age'])
						all[2].append(i)
						all[3].append(data['peer'])
						final[0][data['peer']] = data['metrics']['accuracy']
						final[1][data['peer']] = data['age']
					if data['action'] == 'communicate':
						for r in data['sending_to']:
							diversity[r][data['peer']] = True
				except (json.JSONDecodeError, KeyError):
					# Skip lines that don't contain the expected JSON structure
					continue
	return (all, final, diversity)

(all, final, diversity) = extract_data(sys.argv[1])
diversity_sum = [r.count(True) for r in diversity]

# Create the plots
fig, ax = plt.subplots()
scatter = ax.scatter(all[1], all[0], marker='o', c=all[2], cmap='viridis')
ax.set_ylim(ymin=0, ymax=1)
ax.set_xlabel('Age')
ax.set_ylabel('Accuracy')
cax = ax.inset_axes((1.02, 0, 0.02, 1))
fig.colorbar(scatter, cax=cax, label='Wall Time', ticks=[], aspect=50)
# plt.title('Accuracy vs Age')

# Zoomed inset for scenario 1
# axins = zoomed_inset_axes(ax, 5, axes_kwargs={'aspect': 1167, 'anchor': "SE"}, loc="lower right")
# # As lines per peer
# by_peer = {i: {'acc': [], 'age': []} for i in range(NUM_PEERS)}
# for i in range(len(all[3])):
# 	p = all[3][i]
# 	by_peer[p]['acc'].append(all[0][i])
# 	by_peer[p]['age'].append(all[1][i])
# for i in range(NUM_PEERS):
# 	axins.plot(by_peer[i]['age'], by_peer[i]['acc'])
# As scatter
# axins.scatter(all[1], all[0], marker='o', c=all[2], cmap='viridis')

# axins.set_xlim(0, 700)
# axins.set_ylim(0.5, 0.8)
# axins.tick_params(size=0, labelleft=False, labelbottom=False)
# mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

# Zoomed inset for scenario 2
# axins = zoomed_inset_axes(ax, 16, axes_kwargs={'aspect': 52500, 'anchor': "SE"}, loc="lower right")
# axins.scatter(all[1], all[0], marker='o', c=all[2], cmap='viridis')
# axins.scatter(final[1], final[0], marker='.', c='k')
# axins.set_xlim(750, 3900)
# axins.set_ylim(0.72, 0.75)
# axins.tick_params(size=0, labelleft=False, labelbottom=False)
# mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

# Second scatter plot: Last Accuracy vs Age as overlay
ax.scatter(final[1], final[0], marker='.', c='k')

if len(sys.argv) > 2:
	ref_all, _, _ = extract_data(sys.argv[2])
	ax.plot(ref_all[1], ref_all[0], c='tab:orange')

plt.tight_layout()

def print_stats(data):
	print({k: v for k, v in zip(range(NUM_PEERS), data)})
	s = sorted(data)
	print("Sorted: {}".format(s))
	print("High: {}; Low: {}; Mean: {}; Median: {}\n".format(s[-1], s[0], sum(s)/len(s), s[len(s)//2]))

# Show the plot
try:
	print("## Diversity -----------------------------")
	print_stats(diversity_sum)

	print("## Age -----------------------------------")
	print_stats(final[1])

	print("## Accuracy ------------------------------")
	print_stats(final[0])

	print("## Correlation ---------------------------")
	corr = np.corrcoef(final[0], final[1])
	print("Accuracy and age (final): {}".format(corr[0][1]))
	corr = np.corrcoef(final[0], diversity_sum)
	print("Accuracy and diversity (final): {}".format(corr[0][1]))
	corr = np.corrcoef(final[1], diversity_sum)
	print("Age and diversity (final): {}".format(corr[0][1]))

	# PDF output, no dependencies
	plt.savefig(fname="scatter.pdf", format="pdf", dpi=300)
	# Interactive GUI, requires tornado
	plt.show()
except KeyboardInterrupt:
	plt.close('all')
	exit(0)
