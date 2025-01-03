import numpy as np
import pandas as pd
import argparse
import time
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt
import torch
from scipy.sparse import csr_matrix

from matplotlib.pyplot import cm
from matplotlib.lines import Line2D


from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine


##draw a simple 2D graph using networkx. draw_simple_graph(obj<Graphs>*, int<graph_id>*, str<savefinename>)
def draw_simple_graph(gt, graph_id, filename='plotgraph.png'):
    colorslist =cm.rainbow(np.linspace(0, 1, len(gt.edge_type_id)))
    G = nx.MultiDiGraph()
    pos = nx.spring_layout(G)
    arc_rad = 0.2

    for node_id in range(0,gt.number_of_graph_nodes[graph_id]):
        for node_edge_num in range(0, gt.graph_node_edge_counter[gt.node_index[graph_id] + node_id]):
            edge_index = gt.edge_index[gt.node_index[graph_id] + node_id] + node_edge_num
            G.add_edge(str(node_id), str(gt.edge[edge_index][0]), weight=gt.edge[edge_index][1])

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    legend_elements=[]

    for k in range(len(gt.edge_type_id)):
        eset = [(u, v) for (u, v, d) in G.edges(data=True) if int(d["weight"]) == k]
        elabls = [d for (u, v, d) in G.edges(data=True) if int(d["weight"]) == k]
        le = 'Unknown'
        for ln in gt.edge_type_id.keys():
            if gt.edge_type_id[ln] == k:
                le = ln
                break
    legend_elements.append(Line2D([0], [0], marker='o', color=colorslist[k], label=le, lw=0))

    nx.draw_networkx_edges(G, pos, edgelist=eset, edge_color = colorslist[k], connectionstyle=f'arc3, rad = {arc_rad*k}')
    print(legend_elements)
    plt.title('Graph '+str(graph_id))
    plt.legend(handles=legend_elements, loc='upper left')
    plt.savefig(filename, dpi=300, bbox_inches='tight')


##print the nodes, seperated by (). show_graph_nodes(obj<Graphs>*, int<graph_id>*)
def show_graph_nodes(gt, graph_id):
    graphstr ='Graph#'+str(graph_id)+':\n'
    for node_id in range(gt.number_of_graph_nodes[graph_id]):
        nodestr='Node#'+str(node_id)+'('
        for (symbol_name, symbol_id) in gt.symbol_id.items():
            match = True
            for k in gt.hypervectors[symbol_id,:]:
                chunk = k // 32
                pos = k % 32

    if (gt.X[gt.node_index[graph_id] + node_id][chunk] & (1 << pos)) == 0:
        match = False

    if match:
        nodestr+= symbol_name+' '
    else:
        nodestr+= '*'+' '
        nodestr+= ')'+' '
        graphstr+= nodestr
        print(graphstr)

##print the edges, grouped by source node. show_graph_edges(obj<Graphs>*, int<graph_id>*)
def show_graph_edges(gt, graph_id):
    graphstr ='Graph#'+str(graph_id)+':\n'
    for node_id in range(0,gt.number_of_graph_nodes[graph_id]):
        for node_edge_num in range(0, gt.graph_node_edge_counter[gt.node_index[graph_id] + node_id]):
            edge_index = gt.edge_index[gt.node_index[graph_id] + node_id] + node_edge_num
            edgestr='Edge#'+str(edge_index)

            edgestr+= ' SrcNode#'+str(node_id)

            edgestr+= ' DestNode#'+str(gt.edge[edge_index][0])

            edgestr+= ' Type#'+str(gt.edge[edge_index][1])+'\n'

            graphstr+= edgestr
            graphstr+= '\n'
            print(graphstr)



def dataloader(path):
    data = pd.read_csv(path)
    X = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    #Checking data split
    x_win = (y==0).sum()
    O_win = (y==1).sum()
    print(f'0: {x_win}')
    print(f'1: {O_win}')
    
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    print(f"X_train shape: {X.shape}")
    print(f"X_test shape: {X_test.shape}")
    return X, X_test, y, y_test

def gen_matrix(size):
    nodes = size * size
    adjacency_matrix = np.zeros((nodes, nodes), dtype=int)

    def index(row, col):
        return (row - 1) * size + (col - 1)

    for row in range(1, size + 1):
        for col in range(1, size + 1):
            current_index = index(row, col)
            if row > 1:
                adjacency_matrix[current_index, index(row - 1, col)] = 1
            if row < size: 
                adjacency_matrix[current_index, index(row + 1, col)] = 1
            if col > 1: 
                adjacency_matrix[current_index, index(row, col - 1)] = 1
            if col < size:
                adjacency_matrix[current_index, index(row, col + 1)] = 1
            if row > 1 and col < size: 
                adjacency_matrix[current_index, index(row - 1, col + 1)] = 1
            if row < size and col > 1: 
                adjacency_matrix[current_index, index(row + 1, col - 1)] = 1

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
    
def hypervectors(symbols, connections, board_size):
        moves = []

        for i, symbol in enumerate(symbols):
            if symbol in ["X", "O"]:
                row, col = divmod(i, board_size)

                if row == 0 and (col == 0 or col == board_size - 1):
                    quadrant = "A"
                elif row == 0: 
                    quadrant = "B"
                elif row == 1 and (col == 0 or col == board_size - 1):
                    quadrant = "B"
                elif row == 1:
                    quadrant = "D" if col in {2, 3} else "C"
                elif row in {2, 3}:
                    if col < 2:
                        quadrant = "C"
                    elif col in {2, 3}:
                        quadrant = "E" if row == 2 else "H"
                    else:
                        quadrant = "H"
                elif row == 4: 
                    if col < 2:
                        quadrant = "C"
                    elif col in {2, 3}:
                        quadrant = "D"
                    else:
                        quadrant = "F"
                elif row == 5:
                    quadrant = "G" if col == 0 or col == board_size - 1 else "F"                
                num_connections = connections[i]
                moves.append(f"{symbol}{quadrant}{num_connections}")
            else:
                moves.append(" ")
        #print(moves)
        return moves

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
    parser.add_argument("--epochs", default=2500, type=int)
    parser.add_argument("--number-of-clauses", default=5000, type=int)
    parser.add_argument("--T", default=30, type=int)
    parser.add_argument("--s", default=1, type=float)
    #parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=8, type=int)
    parser.add_argument("--hypervector-size", default=512, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=512, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=True, action='store_true')
    parser.add_argument("--max-included-literals", default=32, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

#load data
args = default_args()

board_size = 13

X, X_test, y, y_test = dataloader('13x13.csv')

adjacency_matrix = gen_matrix(board_size)
print(adjacency_matrix)

edges = [np.sum(adjacency_matrix[i]) for i in range(adjacency_matrix.shape[0])]
print(f'Neighbors///edges: {edges}')
#print(torch.cuda.is_available())

symbols = []

for board in range(X.shape[0]):
        sym = X[board]
        vectors = hypervectors(sym, edges, board_size)
        for x in vectors:
            if x not in symbols:
                symbols.append(x)

#print(symbols)
#graphs training data
print("Creating training data")
graphs_train = Graphs(
    number_of_graphs=X.shape[0],
    symbols=symbols,
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
    board = X[graph_id]
    #print(board)
    vectors = hypervectors(board, edges, board_size)
    #print(possible_moves)
    for k in range(board_size ** 2):
        sym = vectors[k]
        graphs_train.add_graph_node_property(graph_id, k, sym)

    for i in range(board_size ** 2):
        for j in range(board_size ** 2):
            if adjacency_matrix[i, j] == 1:
                graphs_train.add_graph_node_edge(graph_id, i, j, edge_type_name=0)
graphs_train.encode()
#show_graph_edges(graphs_train, 1)
#draw_simple_graph(graphs_train, 1)

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
    board = X_test[graph_id]
    vectors = hypervectors(board, edges, board_size)
    for k in range(board_size ** 2):
        sym =vectors[k]
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