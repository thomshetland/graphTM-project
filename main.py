from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from src.utils.x_builder import build_boards_from_moves
from src.utils.utils import transform_dataset, build_symbol_list, boards_to_games_dict, build_hex_adjacency

import numpy as np
from sklearn.model_selection import train_test_split
from time import time

from config import config

if __name__ == "__main__":

    print("Loading dataset...")
    dataset = np.load("dataset/hex_5x5_5000.npz")
    moves = dataset["moves"]
    lengths = dataset["lengths"]
    y_ds = dataset["winners"]
    x_ds = build_boards_from_moves(moves, lengths, offset=0)

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
    print(f"THIS IS GRAPH LENGTH {train_graph_length}")    

    print("Converting board to game disctionaries")
    train_games = boards_to_games_dict(X_train, config.game.board_size)
    test_games = boards_to_games_dict(X_test, config.game.board_size)

    print("Creating nodes and edges")
    edges = build_hex_adjacency(config.game.board_size)

    print("Creating training graphs")
    print("Creating symbols")
    symbols = build_symbol_list(config.game.board_size)

    # Adjust hypervector size and bits
    edge_symbols = config.edge.symbols.copy()
    print(edge_symbols)

    graphs_train = Graphs(
        train_graph_length,
        symbols=symbols,
        hypervector_size=len(symbols),
        hypervector_bits=config.model.hypervector_bits
    )
    
    for graph_id in range(train_graph_length):
        graphs_train.set_number_of_graph_nodes(graph_id, (config.game.board_size ** 2) + 4)
    
    print("Preparing node configuration")
    graphs_train.prepare_node_configuration()

    n_board = config.game.board_size**2
    for graph_id in range(train_graph_length):
        for i in range(config.game.board_size):
            for j in range(config.game.board_size):
                node_id = i * config.game.board_size + j
                node = (i, j)

                degree = len(edges[node])

                '''# Extra edges for sides (self-loops)
                side_degree = 0
                if j == 0:
                    side_degree += 1          # Left Side
                if j == config.game.board_size - 1:
                    side_degree += 1          # Right Side
                if i == 0:
                    side_degree += 1          # Top Side
                if i == config.game.board_size - 1:
                    side_degree += 1          # Bottom Side'''

                graphs_train.add_graph_node(
                    graph_id,
                    node_id,
                    degree #+ side_degree
                )
        # add virtual nodes
        for i in range(4):
            node_id = n_board + i
            graphs_train.add_graph_node(
                graph_id,
                node_id,
                config.game.board_size
            )

    print("Preparing edge configuration")
    graphs_train.prepare_edge_configuration()
    edge_type_same = {
        1: "1",  # player1-player1
        2: "2",  # player2-player2
    }
    for graph_id in range(train_graph_length):
        game = train_games[graph_id]
        for i in range(config.game.board_size):
            for j in range(config.game.board_size):
                node_id = i * config.game.board_size + j
                node = (i, j)
                cell_value = game.get(node_id, 0)

                for neighbor in edges[node]:
                    ni, nj = neighbor
                    if ni < 0:
                        if ni == -2 and nj == -1: # left
                            neighbor_id = n_board
                            cell_neighbor = 1
                        elif ni == -1 and nj == -2: # right
                            neighbor_id = n_board + 1
                            cell_neighbor = 1
                        elif ni == -1 and nj == -1: # top
                            neighbor_id = n_board + 2 
                            cell_neighbor = 2
                        else:                       # bottom
                            neighbor_id = n_board + 3
                            cell_neighbor = 2
                        
                        if cell_value == cell_neighbor and cell_value != 0:
                            edge_label = edge_type_same[cell_value]
                        else:
                            edge_label = "Plain"
                            #print("Added plain edge with virtual node")
                        graphs_train.add_graph_node_edge(
                            graph_id, 
                            node_id, 
                            neighbor_id, 
                            edge_label
                        )
                        graphs_train.add_graph_node_edge(
                            graph_id, 
                            neighbor_id, 
                            node_id,
                            edge_label
                        )
                    else:
                        neighbor_id = ni * config.game.board_size + nj           
                        cell_neighbor = game.get(neighbor_id, 0)
                        if cell_value == cell_neighbor and cell_value != 0:
                            edge_label = edge_type_same[cell_value]
                            #print(f"CELL VALUE: {cell_value} AND CELL NEIGH: {cell_neighbor}")
                        else:
                            edge_label = "Plain"
                            #print("Added plain edge with real node")

                        graphs_train.add_graph_node_edge(
                            graph_id, 
                            node_id, 
                            neighbor_id, 
                            edge_label
                        )

    #graphs_train.print_graph_edges(3999)
    cell_value_mapping = {
        0: "Empty",
        1: "Player1",   # player1
        2: "Player2",  # player2
    }

    print("Adding training node properties")
    for graph_id in range(train_graph_length):
        game = train_games[graph_id]
        for i in range(config.game.board_size):
            for j in range(config.game.board_size):
                node_id = i * config.game.board_size + j
                cell_value = game.get(node_id, 0)
                cell_property = cell_value_mapping[cell_value]

                graphs_train.add_graph_node_property(
                    graph_id, node_id, f"{cell_property}_{i}_{j}"
                )

                graphs_train.add_graph_node_property(
                    graph_id, node_id, f"c{i+1}_{i}_{j}"
                )
                graphs_train.add_graph_node_property(
                    graph_id, node_id, f"r{j+1}_{i}_{j}"
                )
                num_same = 0
                node = (i, j)
                for neighbor in edges[node]:
                    ni, nj = neighbor
                    neighbor_id = ni * config.game.board_size + nj
                    cell_neighbor = game.get(neighbor_id, 0)
                    if cell_neighbor == cell_value and cell_value != 0:
                        num_same += 1

                if num_same > 0:
                    graphs_train.add_graph_node_property(
                        graph_id, node_id, f"Connected_{i}_{j}"
                    )
                
    print("Encoding training graphs")

    graphs_train.encode()

    graphs_test = Graphs(
        test_graph_length,
        symbols=symbols,
        hypervector_size=len(symbols),
        hypervector_bits=config.model.hypervector_bits
    )
    
    for graph_id in range(test_graph_length):
        graphs_test.set_number_of_graph_nodes(graph_id, (config.game.board_size ** 2) + 4)
    
    print("Preparing node configuration")
    graphs_test.prepare_node_configuration()

    n_board = config.game.board_size**2
    for graph_id in range(test_graph_length):
        for i in range(config.game.board_size):
            for j in range(config.game.board_size):
                node_id = i * config.game.board_size + j
                node = (i, j)

                degree = len(edges[node])

                '''# Extra edges for sides (self-loops)
                side_degree = 0
                if j == 0:
                    side_degree += 1          # Left Side
                if j == config.game.board_size - 1:
                    side_degree += 1          # Right Side
                if i == 0:
                    side_degree += 1          # Top Side
                if i == config.game.board_size - 1:
                    side_degree += 1          # Bottom Side'''

                graphs_test.add_graph_node(
                    graph_id,
                    node_id,
                    degree #+ side_degree
                )
        # add virtual nodes
        for i in range(4):
            node_id = n_board + i
            graphs_test.add_graph_node(
                graph_id,
                node_id,
                config.game.board_size
            )

    print("Preparing edge configuration")
    graphs_test.prepare_edge_configuration()
    edge_type_same = {
        1: "1",  # player1-player1
        2: "2",  # player2-player2
    }
    for graph_id in range(test_graph_length):
        game = test_games[graph_id]
        for i in range(config.game.board_size):
            for j in range(config.game.board_size):
                node_id = i * config.game.board_size + j
                node = (i, j)
                cell_value = game.get(node_id, 0)

                for neighbor in edges[node]:
                    ni, nj = neighbor
                    if ni < 0:
                        if ni == -2 and nj == -1: # left
                            neighbor_id = n_board
                            cell_neighbor = 1
                        elif ni == -1 and nj == -2: # right
                            neighbor_id = n_board + 1
                            cell_neighbor = 1
                        elif ni == -1 and nj == -1: # top
                            neighbor_id = n_board + 2 
                            cell_neighbor = 2
                        else:                       # bottom
                            neighbor_id = n_board + 3
                            cell_neighbor = 2
                        
                        if cell_value == cell_neighbor and cell_value != 0:
                            edge_label = edge_type_same[cell_value]
                        else:
                            edge_label = "Plain"
                            #print("Added plain edge with virtual node")
                        graphs_train.add_graph_node_edge(
                            graph_id, 
                            node_id, 
                            neighbor_id, 
                            edge_label
                        )
                        graphs_train.add_graph_node_edge(
                            graph_id, 
                            neighbor_id, 
                            node_id,
                            edge_label
                        )
                    else:
                        neighbor_id = ni * config.game.board_size + nj           
                        cell_neighbor = game.get(neighbor_id, 0)
                        if cell_value == cell_neighbor and cell_value != 0:
                            edge_label = edge_type_same[cell_value]
                            #print(f"CELL VALUE: {cell_value} AND CELL NEIGH: {cell_neighbor}")
                        else:
                            edge_label = "Plain"
                            #print("Added plain edge with real node")

                        graphs_train.add_graph_node_edge(
                            graph_id, 
                            node_id, 
                            neighbor_id, 
                            edge_label
                        )

    #graphs_train.print_graph_edges(3999)
    cell_value_mapping = {
        0: "Empty",
        1: "Player1",   # player1
        2: "Player2",  # player2
    }

    print("Adding training node properties")
    for graph_id in range(train_graph_length):
        game = train_games[graph_id]
        for i in range(config.game.board_size):
            for j in range(config.game.board_size):
                node_id = i * config.game.board_size + j
                cell_value = game.get(node_id, 0)
                cell_property = cell_value_mapping[cell_value]

                graphs_train.add_graph_node_property(
                    graph_id, node_id, f"{cell_property}_{i}_{j}"
                )

                graphs_train.add_graph_node_property(
                    graph_id, node_id, f"c{i+1}_{i}_{j}"
                )
                graphs_train.add_graph_node_property(
                    graph_id, node_id, f"r{j+1}_{i}_{j}"
                )
                num_same = 0
                node = (i, j)
                for neighbor in edges[node]:
                    ni, nj = neighbor
                    neighbor_id = ni * config.game.board_size + nj
                    cell_neighbor = game.get(neighbor_id, 0)
                    if cell_neighbor == cell_value and cell_value != 0:
                        num_same += 1

                if num_same > 0:
                    graphs_train.add_graph_node_property(
                        graph_id, node_id, f"Connected_{i}_{j}"
                    )
                
    print("Encoding training graphs")

    graphs_train.encode()
    tm = MultiClassGraphTsetlinMachine(
        config.model.number_of_clauses,
        config.model.T,
        config.model.s,
        depth=config.model.depth,
        message_size=config.vector.msg_size,
        message_bits=config.vector.msg_bits,
        grid=(16*13,1,1),
        block=(128,1,1)
    )

    train_acc = []
    epoch_list = []
    epoch = 0

    print("Starting training...")
    for i in range(config.model.epochs):
        epoch += 1
        start_training = time()

        tm.fit(graphs_train, y_train, epochs=1, incremental=True)

        stop_training = time()

        #start_testing = time()
        #result_test = 100.0 * (tm.predict(graphs_test) == y_test).mean()
        #stop_testing = time()

        result_train = 100.0 * (tm.predict(graphs_train) == y_train).mean()

        train_acc.append(result_train)
        #test_acc.append(result_test)
        epoch_list.append(epoch)

        print(
            f"Epoch: {epoch}, "
            f"Train acc: {result_train:.2f}%, "
            #f"Test acc: {result_test:.2f}%, "
            #f"Epoch time: {stop_testing - start_training:.2f}s"
        )

    print("Training finished.")