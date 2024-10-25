import json, sys
import networkx as nx
import matplotlib.pyplot as plt
NUM_PEERS = 30

def prepare_directed_graph(G, file, highpass=0):
    edges = []
    with open(file, 'r') as file:
        for i, line in enumerate(file):
            # Split the line and extract the JSON part
            parts = line.split(':', 2)
            if len(parts) == 3:
                try:
                    # Parse the JSON data
                    data = json.loads(parts[2].strip().replace("'", '"'))
                    if data['age'] < highpass:
                        continue

                    if data['action'] == 'communicate':
                        for r in data['sending_to']:
                            G.add_edge(data['peer'], r)
                            edges.append({'src': data['peer'], 'dest': r})
                except (json.JSONDecodeError, KeyError):
                    # Skip lines that don't contain the expected JSON structure
                    continue
    print("nodes:", G.number_of_nodes())
    print("edges:", G.number_of_edges())
    # Create a figure with a black background
    plt.figure(figsize=(10, 8))
    
    # Create the layout for the nodes
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the graph elements
    nx.draw(G, pos,
           node_color='lightblue',
           node_size=500,
           arrowsize=20,
           edge_color='grey',
           with_labels=True,
           font_size=12,
           font_weight='bold',
           arrows=True)
    
    # Remove axes
    plt.axis('off')
    
    # Add title
    plt.title("Directed Graph Visualization", color='white', pad=20)
    
    return plt

G = nx.DiGraph()
plt = prepare_directed_graph(G, sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
try:
    plt.show()
except KeyboardInterrupt:
	plt.close('all')
	exit(0)
