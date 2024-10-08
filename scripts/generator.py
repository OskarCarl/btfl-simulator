"""
Generated by Claude 3.5 Sonnet, adapted by Oskar Carl
"""

import random

def generate_play(num_peers, num_steps, max_fits, num_rotate, reduce_randomness=False):
	peers = list(range(num_peers))
	fit_counts = {peer: 0 for peer in peers}
	comm_counts = {peer: 0 for peer in peers}
	if reduce_randomness:
		actions = ['communicate', 'communicate', 'eval']
	else:
		actions = ['communicate', 'communicate', 'rotate', 'eval']
	play = []

	for peer in range(num_peers):
		play.append(f"{peer},rotate,{num_rotate}")

	for _ in range(num_steps - num_peers):
		peer = random.choice(peers)
		if reduce_randomness and comm_counts[peer] > 20:
			actions_temp = actions + ['rotate']
		else:
			actions_temp = actions

		if fit_counts[peer] < max_fits and not (reduce_randomness and comm_counts[peer] < 10):
			actions_temp = actions_temp + ['fit']

		if fit_counts[peer] == 0:
			action = 'fit'
		else:
			action = random.choice(actions_temp)

		if action == 'fit':
			fit_counts[peer] += 1
			play.append(f"{peer},fit")
		elif action == 'rotate':
			comm_counts[peer] = 0
			num_rotate = 2 #random.choice([2, 3])
			play.append(f"{peer},rotate,{num_rotate}")
		elif reduce_randomness and action == 'communicate':
			comm_counts[peer] += 1
			play.append(f"{peer},{action}")
		else:
			play.append(f"{peer},{action}")

	return play

# Generate the play
num_peers = 30
num_neighbors = 2
num_steps = 10000
max_fits = 10

play = generate_play(num_peers, num_steps, max_fits, num_neighbors, True)

with open('output.csv', 'w') as f:
	f.write('peer,action,num\n')
	for p in play:
		f.write(p + '\n')

# Print the first 20 and last 20 lines of the play
print("peer,action,num")
print("\n".join(play[:20]))
print("...")
print("\n".join(play[-20:]))
print(f"\nTotal steps: {len(play)}")
