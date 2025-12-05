from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from src.utils.x_builder import build_boards_from_moves
from src.utils.utils import transform_dataset

import numpy as np
from sklearn.model_selection import train_test_split
from time import time
import csv


def build_parity_symbols():
    """
    We don't need bits here, just four symbols:

        p1_even, p1_odd, p2_even, p2_odd

    Each game/graph will have:
        - P1 node with either p1_even or p1_odd
        - P2 node with either p2_even or p2_odd
    """
    symbols = ["p1_even", "p1_odd", "p2_even", "p2_odd"]
    return symbols


def add_parity_properties_for_board(graphs, graph_id, board):
    """
    Given a board, count stones for each player,
    compute parity, and set the corresponding symbols on P1 and P2 nodes.
    """
    flat = board.reshape(-1)

    p1_count = int(np.sum(flat == 1))
    p2_count = int(np.sum(flat == 2))

    # Parity: 0 = even, 1 = odd
    p1_parity = p1_count & 1
    p2_parity = p2_count & 1

    # P1 node symbol
    if p1_parity == 0:
        graphs.add_graph_node_property(graph_id, "P1", "p1_even")
    else:
        graphs.add_graph_node_property(graph_id, "P1", "p1_odd")

    # P2 node symbol
    if p2_parity == 0:
        graphs.add_graph_node_property(graph_id, "P2", "p2_even")
    else:
        graphs.add_graph_node_property(graph_id, "P2", "p2_odd")


if __name__ == "__main__":
    epochs = 300
    clauses = 5000
    T = 3125
    s = 1.0
    depth = 2

    hv_bits = 1
    hv_size = 32 

    msg_bits = 32
    msg_size = 64

    board_size = 40
    offset = 5  

    print("Loading dataset...")
    dataset = np.load("dataset/hex_40x40_10000.npz")
    moves = dataset["moves"]
    lengths = dataset["lengths"]
    winners = dataset["winners"]  

    print(f"Building boards from moves with offset={offset}...")
    x_ds = build_boards_from_moves(moves, lengths, offset=offset)
    print("Boards shape:", x_ds.shape)

    print("Train/test split")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        x_ds, winners, test_size=0.2, random_state=42
    )

    print("Transforming dataset (if needed)")
    X_train = transform_dataset(X_train_raw)  
    X_test = transform_dataset(X_test_raw)
    print(f"Example transformed board:\n{X_train[0]}")

    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)

    train_graph_length = X_train.shape[0]
    test_graph_length = X_test.shape[0]

    print("Creating parity symbols")
    symbols = build_parity_symbols()
    print("Symbols:", symbols)

    graphs_train = Graphs(
        train_graph_length,
        symbols=symbols,
        hypervector_size=hv_size,
        hypervector_bits=hv_bits,
    )

    for gid in range(train_graph_length):
        graphs_train.set_number_of_graph_nodes(gid, 2)

    print("Preparing training node configuration")
    graphs_train.prepare_node_configuration()

    for gid in range(train_graph_length):
        graphs_train.add_graph_node(gid, "P1", 1)
        graphs_train.add_graph_node(gid, "P2", 1)

    print("Preparing training edge configuration")
    graphs_train.prepare_edge_configuration()

    for gid in range(train_graph_length):
        graphs_train.add_graph_node_edge(gid, "P1", "P2", "Plain")
        graphs_train.add_graph_node_edge(gid, "P2", "P1", "Plain")

    print("Adding parity properties to training graphs")
    for gid in range(train_graph_length):
        board = X_train[gid]  # (board_size, board_size)
        add_parity_properties_for_board(graphs_train, gid, board)

    print("Encoding training graphs")
    graphs_train.encode()

    graphs_test = Graphs(
        test_graph_length,
        init_with=graphs_train,
    )

    for gid in range(test_graph_length):
        graphs_test.set_number_of_graph_nodes(gid, 2)

    print("Preparing test node configuration")
    graphs_test.prepare_node_configuration()

    for gid in range(test_graph_length):
        graphs_test.add_graph_node(gid, "P1", 1)
        graphs_test.add_graph_node(gid, "P2", 1)

    print("Preparing test edge configuration")
    graphs_test.prepare_edge_configuration()

    for gid in range(test_graph_length):
        graphs_test.add_graph_node_edge(gid, "P1", "P2", "Plain")
        graphs_test.add_graph_node_edge(gid, "P2", "P1", "Plain")

    print("Adding parity properties to test graphs")
    for gid in range(test_graph_length):
        board = X_test[gid]
        add_parity_properties_for_board(graphs_test, gid, board)

    print("Encoding test graphs")
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

    print("Starting training (parity-only features, real winners as labels)...")
    for epoch in range(1, epochs + 1):
        start_training = time()
        tm.fit(graphs_train, y_train, epochs=1, incremental=True)
        training_time = time() - start_training

        start_testing = time()
        y_pred_test = tm.predict(graphs_test)
        inference_time = time() - start_testing

        y_pred_train = tm.predict(graphs_train)

        train_acc = 100.0 * (y_pred_train == y_train).mean()
        test_acc = 100.0 * (y_pred_test == y_test).mean()

        print(
            f"Epoch: {epoch}, "
            f"Train acc: {train_acc:.2f}%, "
            f"Test acc: {test_acc:.2f}%, "
            f"Training time: {training_time:.2f}s, "
            f"Inference time: {inference_time:.2f}s"
        )

    print("Training finished.")
