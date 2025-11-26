from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from src.utils.x_builder import build_boards_from_moves
from src.utils.utils import transform_dataset, build_symbol_list, boards_to_games_dict, build_hex_adjacency, build_graphs

import numpy as np
from sklearn.model_selection import train_test_split
from time import time
import csv

if __name__ == "__main__":
    # Prepare CSV logging
    csv_filename = "7x7_performance_2_offset.csv"
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_accuracy", "test_accuracy", "training_time", "inference_time"])


    epochs = 300
    clauses = 5000
    T = 3250
    s = 1.15
    depth = 2
    hv_bits = 1
    hv_size = 1
    msg_bits = 32
    msg_size = 512
    board_size = 8
    n_board = board_size ** 2

    print("Loading dataset...")
    dataset = np.load("dataset/hex_7x7_100000.npz")
    moves = dataset["moves"]
    lengths = dataset["lengths"]
    y_ds = dataset["winners"]
    x_ds = build_boards_from_moves(moves, lengths, offset=2)

    print("Final X shape:", x_ds.shape)

    print("Pre-processing training and test dataset")
    # Split 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        x_ds, y_ds, test_size=0.2, random_state=42
    )

    print(X_train[0])

    print("Transforming dataset")
    X_train = transform_dataset(X_train)
    X_test = transform_dataset(X_test)
    print(f"Transformed dataset to {X_train[0]}")

    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)

    train_graph_length = X_train.shape[0]
    test_graph_length = X_test.shape[0]
      

    print("Converting board to game disctionaries")
    train_games = boards_to_games_dict(X_train, board_size)
    test_games = boards_to_games_dict(X_test, board_size)

    print("Creating nodes and edges")
    edges = build_hex_adjacency(board_size)
    
    print("Creating symbols")
    symbols = build_symbol_list(board_size)
    edge_symbols = ["Plain", "Player 1", "Player 2", "Boarder"]
    symbols.extend(edge_symbols)
    hv_size = len(symbols)
    graphs_train = build_graphs("Train", X=X_train, games=train_games, symbols=symbols, edges=edges, board_size=board_size,hv_size=hv_size, hv_bits=hv_bits)
    graphs_test = build_graphs("Test", graphs_train, X_test, test_games, symbols, edges, board_size, hv_size, hv_bits)

    tm = MultiClassGraphTsetlinMachine(
        clauses,
        T,
        s,
        depth=depth,
        message_size=msg_size,
        message_bits=msg_bits,
        grid=(16*13,1,1),
        block=(128,1,1)
    )

    train_acc = []
    test_acc = []
    epoch_list = []
    epoch = 0
    print("Starting training...")

    for i in range(epochs):
        epoch += 1

        # ---- Training ----
        start_training = time()
        tm.fit(graphs_train, y_train, epochs=1, incremental=True)
        stop_training = time()
        training_time = stop_training - start_training

        # ---- Inference ----
        start_testing = time()
        result_test = 100.0 * (tm.predict(graphs_test) == y_test).mean()
        stop_testing = time()
        inference_time = stop_testing - start_testing

        result_train = 100.0 * (tm.predict(graphs_train) == y_train).mean()

        # Store values in lists (optional)
        train_acc.append(result_train)
        test_acc.append(result_test)
        epoch_list.append(epoch)

        # Print to console
        print(
            f"Epoch: {epoch}, "
            f"Train acc: {result_train:.2f}%, "
            f"Test acc: {result_test:.2f}%, "
            f"Training time: {training_time:.2f}s, "
            f"Inference time: {inference_time:.2f}s"
        )

        # ---- Append to CSV ----
        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, result_train, result_test, training_time, inference_time])
