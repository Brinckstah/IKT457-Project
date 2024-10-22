import numpy as np
import pandas as pd
import argparse
import time
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt


from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def dataloader(path):
    data = pd.read_csv(path)
    X = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    #Checking data split
    x_win = (y==0).sum()
    O_win = (y==1).sum()
    print(f'0: {x_win}')
    print(f'1: {O_win}')
    
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=7)

    print(f"X_train shape: {X.shape}")
    print(f"X_test shape: {X_test.shape}")
    return X, X_test, y, y_test

#create adjacency matrix for an NxN grid - Coordinate based
def create_matrix(board_size):
    n_nodes = board_size ** 2  
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    #checks if position is within boundary
    def boundary_check(pos, board_size):
        row, col = pos
        return 0 <= row < board_size and 0 <= col < board_size

    #position to index
    def pto(pos, board_size):
        row, col = pos
        return row * board_size + col

    #possible relative moves
    moves_even = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    moves_odd  = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]

    #loop through each node in the grid
    for row in range(board_size):
        for col in range(board_size):
            current_node = (row, col)
            current_node_index = pto(current_node, board_size)
            
            moves = moves_even if row % 2 == 0 else moves_odd

            for move in moves:
                neighbor = (row + move[0], col + move[1])
                if boundary_check(neighbor, board_size):
                    neighbor_index = pto(neighbor, board_size)
                    adjacency_matrix[current_node_index, neighbor_index] = 1
                    adjacency_matrix[neighbor_index, current_node_index] = 1

    return adjacency_matrix

def generate_edge_count(size):
    num_nodes = size * size
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    def node_index(row, col):
        return (row - 1) * size + (col - 1)

    for row in range(1, size + 1):
        for col in range(1, size + 1):
            current_index = node_index(row, col)
            # Connecting nodes based on hexagonal connections
            if row > 1:  # Connect to the node above
                adjacency_matrix[current_index, node_index(row - 1, col)] = 1
            if row < size:  # Connect to the node below
                adjacency_matrix[current_index, node_index(row + 1, col)] = 1
            if col > 1:  # Connect to the node to the left
                adjacency_matrix[current_index, node_index(row, col - 1)] = 1
            if col < size:  # Connect to the node to the right
                adjacency_matrix[current_index, node_index(row, col + 1)] = 1
            if row > 1 and col < size:  # Connect to the upper right node
                adjacency_matrix[current_index, node_index(row - 1, col + 1)] = 1
            if row < size and col > 1:  # Connect to the lower left node
                adjacency_matrix[current_index, node_index(row + 1, col - 1)] = 1

    return adjacency_matrix

def print_rules():
    weights = tm.get_state()[1].reshape(2, -1)
    for i in range(tm.number_of_clauses):
        print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
        l = []
        for k in range(args.hypervector_size * 2):
            if tm.ta_action(0, i, k):
                if k < args.hypervector_size:
                    l.append("x%d" % (k))
                else:
                    l.append("NOT x%d" % (k - args.hypervector_size))
        print(" AND ".join(l))
        print(f"Number of literals: {len(l)}")
                
def visualize(matrix):
    G = nx.from_numpy_array(matrix)

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=800, font_size=10, font_weight='bold')
    plt.title("Graph Representation of the Adjacency Matrix")
    plt.show()

def visualize_hexagonal_grid(adjacency_matrix, board_size):
    G = nx.Graph()  # Create an empty graph

    # Add nodes
    for node in range(board_size ** 2):
        G.add_node(node)

    # Add edges from the adjacency matrix
    n_nodes = board_size ** 2
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)

    # Create positions for hexagonal layout (manual positioning for hex grid)
    pos = {}
    for row in range(board_size):
        for col in range(board_size):
            index = row * board_size + col
            x = col * 2 + (row % 2)  # Offset odd rows
            y = -row * np.sqrt(3)  # Use a vertical spacing for rows
            pos[index] = (x, y)

    # Plot the hexagonal grid
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=12, font_weight="bold", edge_color="gray")
    plt.title(f"Hexagonal Grid for Board Size {board_size}x{board_size}")
    plt.show()

# Training loop
def training():
    start_training = time.time()
    for i in range(args.epochs):
        tm.fit(graphs_train, y, epochs=1, incremental=True)
        print(f"Epoch#{i+1} -- Accuracy train: {100*(tm.predict(graphs_train) == y).mean()}", end=' ')
        print(f"-- Accuracy test: {100*(tm.predict(graphs_test) == y_test).mean()} ")
    stop_training = time.time()
    print(f"Time: {stop_training - start_training}")

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--number-of-clauses", default=50, type=int)
    parser.add_argument("--T", default=400, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    #parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--hypervector-size", default=256, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=16, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

#load data
args = default_args()

board_size = 4

X, X_test, y, y_test = dataloader('hex_game_results.csv')
#X, X_test, y, y_test = dataloader('3x3_small.csv')

#create adjacency matrix
#adjacency_matrix = create_matrix(board_size)
adjacency_matrix = generate_edge_count(board_size)
print(adjacency_matrix)

edges = [np.sum(adjacency_matrix[i]) for i in range(adjacency_matrix.shape[0])]
print(f'Neighbors///edges: {edges}')
#visualize(adjacency_matrix)
visualize_hexagonal_grid(adjacency_matrix, board_size)

#graphs training data
print("Creating training data")
graphs_train = Graphs(
    number_of_graphs=X.shape[0],
    symbols=["O", "X", " "],
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing=args.double_hashing,
)

# prepare nodes
for graph_id in range(X.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id=graph_id, number_of_graph_nodes=board_size ** 2)
graphs_train.prepare_node_configuration()

# add edges
for graph_id in range(X.shape[0]):
    for k in range(board_size ** 2):
        graphs_train.add_graph_node(graph_id, k, edges[k])
graphs_train.prepare_edge_configuration()

# add edges between connected nodes
for graph_id in range(X.shape[0]):
    for k in range(board_size ** 2):
        sym = X[graph_id][k]
        graphs_train.add_graph_node_property(graph_id, k, sym)
    

    for i in range(board_size ** 2):
        for j in range(board_size ** 2):
            if adjacency_matrix[i, j] == 1:
                graphs_train.add_graph_node_edge(graph_id, i, j, edge_type_name=0)

graphs_train.encode()

#test data preparation
print("Creating test data")
graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)

for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id=graph_id, number_of_graph_nodes=board_size ** 2)
graphs_test.prepare_node_configuration()

for graph_id in range(X_test.shape[0]):
    for k in range(board_size ** 2):
        graphs_test.add_graph_node(graph_id, k, edges[k])
graphs_test.prepare_edge_configuration()

for graph_id in range(X_test.shape[0]):
    for k in range(board_size ** 2):
        sym = X_test[graph_id][k]
        graphs_test.add_graph_node_property(graph_id, k, sym)

    for i in range(board_size ** 2):
        for j in range(board_size ** 2):
            if adjacency_matrix[i, j] == 1:
                graphs_test.add_graph_node_edge(graph_id, i, j, edge_type_name=0)

graphs_test.encode()

# Train the Tsetlin Machine
tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals,
    grid=(16*13,1,1),
    block=(128,1,1)
)

training()

#print_rules()
