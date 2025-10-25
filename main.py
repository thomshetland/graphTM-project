from GraphTsetlinMachine.graphs import Graphs
#from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from src.x_builder import build_boards_from_moves

import numpy as np
import numba
import random
import argparse

if __name__ == "__main__":
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
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=32, type=int)
    args = parser.parse_args()

    dataset = np.load("dataset/hex_11x11_5000.npz")
    moves = dataset["moves"]
    lengths = dataset["lengths"]
    y_ds = dataset["winners"]
    x_ds = build_boards_from_moves(moves, lengths, offset=0)

    
    print("Final X shape:", x_ds.shape)
    '''board_size = 5
    number_of_nodes = board_size**2
    number_of_outgoing_edges = 6
    number_of_games = len(x_train)

    graphs_train = Graphs(
        number_of_nodes, # sets how many graphs to make
        symbols = ['A', 'B', 'corner'], # symbols for each feature (XOR has 2 feature)
        hypervector_size = 32, # how large the hypervector is to store symbols.
        hypervector_bits = 2, # set how many bits to use to represent symbols
    )

    for graph_id in range(number_of_nodes):
        graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)

    graphs_train.prepare_node_configuration()

    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        number_of_state_bits = args.number_of_state_bits,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        double_hashing = args.double_hashing
    )

    for i in range(args.epochs):
        start_training = time()
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
        stop_testing = time()

        result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

        print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))'''