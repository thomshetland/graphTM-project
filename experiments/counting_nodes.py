from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from src.utils.x_builder import build_boards_from_moves
from src.utils.utils import transform_dataset, boards_to_games_dict, build_hex_adjacency

import numpy as np
from sklearn.model_selection import train_test_split
from time import time


def build_count_symbols(board_size):
    n_board = board_size ** 2
    symbols = []

    for c in range(n_board + 1):
        symbols.append(f"stones_p1_{c}")
    for c in range(n_board + 1):
        symbols.append(f"stones_p2_{c}")

    return symbols


if __name__ == "__main__":
    epochs = 300
    clauses = 5000
    T = 3250
    s = 1.15
    depth = 2
    hv_bits = 1
    hv_size = 1
    msg_bits = 32
    msg_size = 1024
    board_size = 7
    n_board = board_size ** 2

    print("Loading dataset...")
    dataset = np.load("dataset/hex_15x15_5000.npz")
    moves = dataset["moves"]
    lengths = dataset["lengths"]
    y_ds = dataset["winners"]

    # Boards from move sequences
    x_ds = build_boards_from_moves(moves, lengths, offset=0)
    print("Final X shape:", x_ds.shape)

    print("Pre-processing training and test dataset")
    # Split 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        x_ds, y_ds, test_size=0.2, random_state=42
    )

    print("Example raw board:", X_train[0])

    print("Transforming dataset")
    X_train = transform_dataset(X_train)
    X_test = transform_dataset(X_test)
    print(f"Transformed dataset example: {X_train[0]}")

    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)

    train_graph_length = X_train.shape[0]
    test_graph_length = X_test.shape[0]

    print("Converting boards to game dictionaries")
    train_games = boards_to_games_dict(X_train, board_size)
    test_games = boards_to_games_dict(X_test, board_size)

    print("Creating adjacency for Hex board")
    edges = build_hex_adjacency(board_size)

    print("Creating symbols based on stone counts only")
    symbols = build_count_symbols(board_size)

    edge_symbols = ["Plain", "Player 1", "Player 2"]
    symbols.extend(edge_symbols)
    print("Edge symbols:", edge_symbols)
    print("Total symbols:", len(symbols))

    graphs_train = Graphs(
        train_graph_length,
        symbols=symbols,
        hypervector_size=len(symbols),
        hypervector_bits=hv_bits,
    )

    n_board = board_size ** 2

    # Only real board nodes
    for graph_id in range(train_graph_length):
        graphs_train.set_number_of_graph_nodes(graph_id, n_board)

    print("Preparing training node configuration")
    graphs_train.prepare_node_configuration()

    # Add board nodes (with degree from hex adjacency, excluding virtual neighbors)
    for graph_id in range(train_graph_length):
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                node = (i, j)
                valid_neighbors = [nbr for nbr in edges[node] if nbr[0] >= 0]
                degree = len(valid_neighbors)

                graphs_train.add_graph_node(graph_id, node_id, degree)

    print("Preparing training edge configuration")
    graphs_train.prepare_edge_configuration()

    edge_type_same = {
        1: "1",  # player1-player1
        2: "2",  # player2-player2
    }

    # Add edges only between real board nodes
    for graph_id in range(train_graph_length):
        game = train_games[graph_id]
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                node = (i, j)
                cell_value = game.get(node_id, 0)

                for neighbor in edges[node]:
                    ni, nj = neighbor

                    # Skip virtual neighbors
                    if ni < 0:
                        continue

                    neighbor_id = ni * board_size + nj
                    cell_neighbor = game.get(neighbor_id, 0)

                    if cell_value == cell_neighbor and cell_value != 0:
                        edge_label = edge_type_same[cell_value]
                    else:
                        edge_label = "Plain"

                    graphs_train.add_graph_node_edge(
                        graph_id, node_id, neighbor_id, edge_label
                    )

    # Add global stone-count properties to each node
    for graph_id in range(train_graph_length):
        game = train_games[graph_id]

        # Count stones for the whole board
        p1_count = sum(1 for v in game.values() if v == 1)
        p2_count = sum(1 for v in game.values() if v == 2)

        sym_p1 = f"stones_p1_{p1_count}"
        sym_p2 = f"stones_p2_{p2_count}"

        # Attach the same global-count symbols to every board node
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j

                graphs_train.add_graph_node_property(graph_id, node_id, sym_p1)
                graphs_train.add_graph_node_property(graph_id, node_id, sym_p2)

    print("Encoding training graphs")
    for graph_id in range(min(5, train_graph_length)):
        print(f"Train graph {graph_id}: node properties attached (stone counts only).")
    graphs_train.encode()

    graphs_test = Graphs(
        test_graph_length,
        init_with=graphs_train, 
    )

    for graph_id in range(test_graph_length):
        graphs_test.set_number_of_graph_nodes(graph_id, n_board)

    print("Preparing test node configuration")
    graphs_test.prepare_node_configuration()

    # Add only real board nodes for test graphs
    for graph_id in range(test_graph_length):
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                node = (i, j)
                valid_neighbors = [nbr for nbr in edges[node] if nbr[0] >= 0]
                degree = len(valid_neighbors)

                graphs_test.add_graph_node(graph_id, node_id, degree)

    print("Preparing test edge configuration")
    graphs_test.prepare_edge_configuration()

    # Add edges only between real board nodes for test graphs
    for graph_id in range(test_graph_length):
        game = test_games[graph_id]
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                node = (i, j)
                cell_value = game.get(node_id, 0)

                for neighbor in edges[node]:
                    ni, nj = neighbor

                    # Skip virtual neighbors
                    if ni < 0:
                        continue

                    neighbor_id = ni * board_size + nj
                    cell_neighbor = game.get(neighbor_id, 0)

                    if cell_value == cell_neighbor and cell_value != 0:
                        edge_label = edge_type_same[cell_value]
                    else:
                        edge_label = "Plain"

                    graphs_test.add_graph_node_edge(
                        graph_id, node_id, neighbor_id, edge_label
                    )

    print("Adding test node properties based on stone counts")

    for graph_id in range(test_graph_length):
        game = test_games[graph_id]

        p1_count = sum(1 for v in game.values() if v == 1)
        p2_count = sum(1 for v in game.values() if v == 2)

        sym_p1 = f"stones_p1_{p1_count}"
        sym_p2 = f"stones_p2_{p2_count}"

        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j

                graphs_test.add_graph_node_property(graph_id, node_id, sym_p1)
                graphs_test.add_graph_node_property(graph_id, node_id, sym_p2)

    print("Encoding test graphs")
    for graph_id in range(min(5, test_graph_length)):
        print(f"Test graph {graph_id}: node properties attached (stone counts only).")
    graphs_test.encode()

    tm = MultiClassGraphTsetlinMachine(
        clauses,
        T,
        s,
        depth=depth,
        message_size=msg_size,
        message_bits=msg_bits,
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    )

    train_acc = []
    test_acc = []
    epoch_list = []
    epoch = 0

    print("Starting training (stone-count features only, no virtual nodes)...")
    for i in range(epochs):
        epoch += 1
        start_training = time()

        tm.fit(graphs_train, y_train, epochs=1, incremental=True)

        stop_training = time()

        start_testing = time()
        result_test = 100.0 * (tm.predict(graphs_test) == y_test).mean()
        stop_testing = time()

        result_train = 100.0 * (tm.predict(graphs_train) == y_train).mean()

        train_acc.append(result_train)
        test_acc.append(result_test)
        epoch_list.append(epoch)

        print(
            f"Epoch: {epoch}, "
            f"Train acc: {result_train:.2f}%, "
            f"Test acc: {result_test:.2f}%, "
            f"Epoch time: {stop_testing - start_training:.2f}s"
        )

    print("Training finished.")
