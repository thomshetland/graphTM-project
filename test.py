from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from src.x_builder import build_boards_from_moves

import numpy as np
import random
import argparse
from sklearn.model_selection import train_test_split
from time import time


def create_bridge(edges, node1, node2):
    """Add an undirected edge between two board coordinates."""
    if node2 not in edges[node1]:
        edges[node1].append(node2)
    if node1 not in edges[node2]:
        edges[node2].append(node1)


def build_hex_edges(board_size):
    """
    Build the adjacency list for a hex board of given size.

    Hex neighbors (i,j) have up to 6 neighbors:
      (i-1, j), (i+1, j),
      (i, j-1), (i, j+1),
      (i-1, j+1), (i+1, j-1)
    """
    edges = {}
    for i in range(board_size):
        for j in range(board_size):
            node = (i, j)
            edges[node] = []

    # Create edges according to hex neighborhood
    for i in range(board_size):
        for j in range(board_size):
            node = (i, j)
            neighbors = [
                (i - 1, j),
                (i + 1, j),
                (i, j - 1),
                (i, j + 1),
                (i - 1, j + 1),
                (i + 1, j - 1),
            ]
            for ni, nj in neighbors:
                if 0 <= ni < board_size and 0 <= nj < board_size:
                    create_bridge(edges, node, (ni, nj))
    return edges


def build_symbol_list(board_size):
    """
    Build the list of symbols (node properties) used in the hypervector
    representation.

    - 'Empty', 'Player1', 'Player2'  : which stone is on the cell
    - 'Row_i', 'Col_j'              : where the cell is on the board
    """
    symbols = ["Empty", "Player1", "Player2"]
    symbols.extend([f"Row_{i}" for i in range(board_size)])
    symbols.extend([f"Col_{j}" for j in range(board_size)])
    return symbols


def build_graphs_from_boards(X, edges, symbols, board_size, args):
    """
    Convert an array of boards into a Graphs object.

    X: ndarray of shape (n_games, board_size, board_size) or
       (n_games, board_size * board_size)
    edges: adjacency dict {(i,j): [(ni,nj), ...]}
    symbols: list of strings
    """
    n_games = X.shape[0]
    number_of_nodes = board_size * board_size

    graphs = Graphs(
        n_games,
        symbols=symbols,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
    )

    # Make sure boards are (board_size, board_size)
    X = X.reshape(n_games, board_size, board_size)

    print("Setting number of nodes for each graph")
    for graph_id in range(n_games):
        graphs.set_number_of_graph_nodes(graph_id, number_of_nodes)

    graphs.prepare_node_configuration()

    # Precompute node names and outdegrees
    node_names = {}
    node_outdegree = {}
    for i in range(board_size):
        for j in range(board_size):
            node_name = f"cell_{i}_{j}"
            node_names[(i, j)] = node_name
            node_outdegree[(i, j)] = len(edges[(i, j)])

    print("Adding nodes to graphs")
    for graph_id in range(n_games):
        for i in range(board_size):
            for j in range(board_size):
                node_name = node_names[(i, j)]
                outdeg = node_outdegree[(i, j)]
                graphs.add_graph_node(graph_id, node_name, outdeg)

    print("Preparing and adding edges")
    graphs.prepare_edge_configuration()
    edge_type = "adjacent"

    for graph_id in range(n_games):
        for i in range(board_size):
            for j in range(board_size):
                node_name = node_names[(i, j)]
                for (ni, nj) in edges[(i, j)]:
                    neighbor_name = node_names[(ni, nj)]
                    graphs.add_graph_node_edge(
                        graph_id, node_name, neighbor_name, edge_type
                    )

    print("Adding node properties (colors + position)")
    for graph_id in range(n_games):
        board = X[graph_id]  # shape (board_size, board_size)
        for i in range(board_size):
            for j in range(board_size):
                node_name = node_names[(i, j)]
                cell_value = board[i, j]

                # Cell color / player
                # Adjust these mappings if your x_builder uses different codes
                if cell_value == 0:
                    graphs.add_graph_node_property(graph_id, node_name, "Empty")
                elif cell_value == 1:
                    graphs.add_graph_node_property(graph_id, node_name, "Player1")
                elif cell_value == 2:
                    graphs.add_graph_node_property(graph_id, node_name, "Player2")
                else:
                    # Just in case of unexpected values
                    graphs.add_graph_node_property(graph_id, node_name, "Empty")

                # Positional properties: which row and column this cell is in
                graphs.add_graph_node_property(graph_id, node_name, f"Row_{i}")
                graphs.add_graph_node_property(graph_id, node_name, f"Col_{j}")

    print("Encoding graphs (binding + bundling hypervectors)")
    graphs.encode()
    return graphs


if __name__ == "__main__":
    print("Initializing program")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--number-of-clauses", default=20000, type=int)
    parser.add_argument("--T", default=25000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=1024, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument(
        "--double-hashing",
        dest="double_hashing",
        default=False,
        action="store_true",
    )
    parser.add_argument("--max-included-literals", default=32, type=int)
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = np.load("dataset/hex_5x5_5000.npz")
    moves = dataset["moves"]
    lengths = dataset["lengths"]
    y_ds = dataset["winners"].astype(np.uint32)

    # x_ds should be an array of shape (n_games, 5, 5) or (n_games, 25)
    x_ds = build_boards_from_moves(moves, lengths, offset=0)
    print("Final X shape:", x_ds.shape)

    print("Pre-processing training and test dataset")
    X_train, X_test, y_train, y_test = train_test_split(
        x_ds, y_ds, test_size=0.2, random_state=42
    )

    board_size = 5
    print("Creating hex edges")
    edges = build_hex_edges(board_size)

    print("Building symbol list")
    symbols = build_symbol_list(board_size)

    print("Building training graphs")
    graphs_train = build_graphs_from_boards(
        X_train, edges, symbols, board_size, args
    )

    print("Building test graphs")
    graphs_test = build_graphs_from_boards(
        X_test, edges, symbols, board_size, args
    )

    print("Initializing MultiClassGraphTsetlinMachine")
    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        number_of_state_bits=args.number_of_state_bits,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        double_hashing=args.double_hashing,
    )

    print("Starting training")
    for epoch in range(args.epochs):
        start_training = time()
        tm.fit(graphs_train, y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        y_pred_test = tm.predict(graphs_test)
        stop_testing = time()

        test_acc = 100.0 * (y_pred_test == y_test).mean()
        train_acc = 100.0 * (tm.predict(graphs_train) == y_train).mean()

        print(
            "%d %.2f %.2f %.2f %.2f"
            % (
                epoch,
                train_acc,
                test_acc,
                stop_training - start_training,
                stop_testing - start_testing,
            )
        )
