"""
Generated by Claude 3.5 Sonnet, adapted by Oskar Carl
"""

import random

def generate_play(num_peers, num_steps, max_fits):
	peers = list(range(num_peers))
	fit_counts = {peer: 0 for peer in peers}
	actions = ['communicate', 'communicate', 'rotate', 'eval']
	play = []

	for _ in range(num_steps):
		peer = random.choice(peers)
		if fit_counts[peer] == 0:
			action = 'fit'
		elif fit_counts[peer] < max_fits:
			action = random.choice(actions + ['fit'])
		else:
			action = random.choice(actions)

		if action == 'fit':
			fit_counts[peer] += 1
			play.append(f"{peer},fit")
		elif action == 'rotate':
			num_neighbors = random.choice([2, 3])
			play.append(f"{peer},rotate,{num_neighbors}")
		else:
			play.append(f"{peer},{action}")

	return play

# Generate the play
num_peers = 30
num_steps = 10000
max_fits = 10

play = generate_play(num_peers, num_steps, max_fits)

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