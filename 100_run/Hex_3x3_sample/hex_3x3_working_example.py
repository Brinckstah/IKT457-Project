import argparse
import numpy as np
import pandas as pd
from time import time
# Graph Tsetlin Machine stuff
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--number-of-clauses", default=200, type=int)
    parser.add_argument("--T", default=400, type=int)
    parser.add_argument("--s", default=1.2, type=float)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--hypervector-size", default=512, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=512, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=16, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

# Load data
data = pd.read_csv('3x3_small.csv')

board_size = 3
subset_size = int(data.shape[0] * 0.9)
test_size = data.shape[0] - subset_size
X = data.iloc[:subset_size, 0].values
X_test = data.iloc[subset_size:subset_size + test_size, 0].values
y = data.iloc[:subset_size, 1].values
y_test = data.iloc[subset_size:subset_size + test_size, 1].values

print(f"X_train shape: {X.shape}")
print(f"X_test shape: {X_test.shape}")

symbol_names = ["O", "X", " "]

# Create a list with node pairs for the edges
edges = []
for i in range(board_size):
    for j in range(1, board_size):
        # Connect rows
        edges.append(((i, j-1), (i, j)))
        # Connect columns
        edges.append(((j-1, i), (j, i)))
        # Connect "back-columns"
        if i < board_size - 1:
            edges.append(((i, j), (i+1, j-1)))


# Make a list with the number of edges for each node
# There is probably a better way to do this. I.e., with an adjacency matrix
n_edges_list = []
for i in range(board_size**2):
    # Top left and bottom right corner have 2 neighbors
    if i == 0 or i == board_size**2-1:
        n_edges_list.append(2)
    # Top right and bottom left corners have 3 neighbors
    elif i == board_size - 1 or i == board_size**2-board_size:
        n_edges_list.append(3)
    # Top and bottom edges excluding corners has 4 neighbors
    elif i // board_size == 0 or i // board_size == board_size - 1:
        n_edges_list.append(4)
    # The side nodes excluding corners also has 4 neighbors
    elif i % board_size == 0 or i % board_size == board_size-1:
        n_edges_list.append(4)
    # The interior nodes has 6 edges
    else:
        n_edges_list.append(6)


# Helper function
def position_to_edge_id(pos, board_size):
    return pos[0] * board_size + pos[1]

print("Creating training data")
graphs_train = Graphs(
    number_of_graphs=subset_size,
    symbols=symbol_names,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing=args.double_hashing,
)

# Prepare nodes
for graph_id in range(X.shape[0]):
    graphs_train.set_number_of_graph_nodes(
        graph_id=graph_id,
        number_of_graph_nodes=board_size**2,
    )
graphs_train.prepare_node_configuration()

print(n_edges_list)
# Prepare edges
for graph_id in range(X.shape[0]):
    for k in range(board_size**2):
        graphs_train.add_graph_node(graph_id, k, n_edges_list[k])
graphs_train.prepare_edge_configuration()

# Create the graph
for graph_id in range(X.shape[0]):
    for k in range(board_size**2):
        sym = X[graph_id][k]
        graphs_train.add_graph_node_property(graph_id, k, sym)
    # Loop through all edges
    for edge in edges:
        node_id = position_to_edge_id(edge[0], board_size)
        destination_node_id = position_to_edge_id(edge[1], board_size)
        graphs_train.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type_name=0)
        graphs_train.add_graph_node_edge(graph_id, destination_node_id, node_id, edge_type_name=0)
graphs_train.encode()

# Test graph
print("Creating test data")
graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)

# Prepare nodes
for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(
        graph_id=graph_id,
        number_of_graph_nodes=board_size**2,
    )
graphs_test.prepare_node_configuration()

# Prepare edges
for graph_id in range(X_test.shape[0]):
    for k in range(board_size**2):
        graphs_test.add_graph_node(graph_id, k, n_edges_list[k])
graphs_test.prepare_edge_configuration()

# Create the graph
for graph_id in range(X_test.shape[0]):
    for k in range(board_size**2):
        sym = X_test[graph_id][k]
        graphs_test.add_graph_node_property(graph_id, k, sym)
         # Add edges
    for edge in edges:
        node_id = position_to_edge_id(edge[0], board_size)
        destination_node_id = position_to_edge_id(edge[1], board_size)
        graphs_test.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type_name=0)
        graphs_test.add_graph_node_edge(graph_id, destination_node_id, node_id, edge_type_name=0)
        
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

start_training = time()
for i in range(args.epochs):
    tm.fit(graphs_train, y, epochs=1, incremental=True)
    print(f"Epoch#{i+1} -- Accuracy train: {np.mean(y == tm.predict(graphs_train))}", end=' ')
    print(f"-- Accuracy test: {np.mean(y_test == tm.predict(graphs_test))} ")
stop_training = time()
print(f"Time: {stop_training - start_training}")


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

