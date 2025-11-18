from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from src.utils.x_builder import build_boards_from_moves
from src.utils.utils import transform_dataset

import numpy as np
from sklearn.model_selection import train_test_split
from time import time

from config import config  # YAML -> config.model, config.edge, config.vector, config.game


def build_hex_adjacency(board_size):
    """Build undirected Hex adjacency on a board_size x board_size grid."""
    edges = {}
    for i in range(board_size):
        for j in range(board_size):
            node = (i, j)
            edges[node] = []

    def connect(a, b):
        if b not in edges[a]:
            edges[a].append(b)

    for i in range(board_size):
        for j in range(board_size):
            node = (i, j)

            if i < board_size - 1:
                connect(node, (i + 1, j))
                connect((i + 1, j), node)

                if j > 0:
                    connect(node, (i + 1, j - 1))
                    connect((i + 1, j - 1), node)

            if j < board_size - 1:
                connect(node, (i, j + 1))
                connect((i, j + 1), node)

            if i > 0:
                connect(node, (i - 1, j))
                connect((i - 1, j), node)

            if j > 0:
                connect(node, (i, j - 1))
                connect((i, j - 1), node)

    return edges


def build_symbol_list(board_size):
    """Our node symbol approach."""
    symbols = []
    for i in range(board_size):
        for j in range(board_size):
            symbols.extend([
                f"Empty_{i}_{j}",
                f"Red_{i}_{j}",
                f"Blue_{i}_{j}",
                f"connected_{i}_{j}",
                f"c{i+1}_{i}_{j}",
                f"r{j+1}_{i}_{j}",
            ])
    return symbols


def boards_to_games_dict(X, board_size):
    """
    X: array-like (n_samples, board_size, board_size) or (n_samples, board_size*board_size)
    Returns: list[dict[node_id -> cell_value]]
    cell_value: 0 (empty), 1 (player1), 2 (player2)
    """
    X_arr = np.asarray(X)
    games = []
    for board in X_arr:
        board_flat = board.reshape(-1)
        game_state = {idx: int(val) for idx, val in enumerate(board_flat)}
        games.append(game_state)
    return games


if __name__ == "__main__":

    # ------------------------------------------------------------
    # 1. Load and prepare dataset
    # ------------------------------------------------------------
    print("Loading dataset...")
    data = np.load("dataset/hex_5x5_5000.npz")
    moves = data["moves"]
    lengths = data["lengths"]
    y_ds = data["winners"]
    x_ds = build_boards_from_moves(moves, lengths, offset=0)

    print("Final X shape (raw):", x_ds.shape)

    print("Pre-processing training and test dataset")
    X_train, X_test, y_train, y_test = train_test_split(
        x_ds, y_ds, test_size=0.2, random_state=42
    )

    print("Example raw board (before transform):")
    print(X_train[0])

    print("Transforming dataset")
    X_train = transform_dataset(X_train)
    X_test = transform_dataset(X_test)

    # 🔧 FIX: make sure these are numpy arrays so .shape works
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    print("Transformed first board to:")
    print(X_train[0])

    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)

    board_size = config.game.board_size
    train_graph_length = X_train.shape[0]   # or len(X_train)
    test_graph_length = X_test.shape[0]

    # ------------------------------------------------------------
    # 2. Build adjacency
    # ------------------------------------------------------------
    print("Creating nodes and edges (adjacency)...")
    edges = build_hex_adjacency(board_size)

    # ------------------------------------------------------------
    # 3. Node symbols + edge symbols
    # ------------------------------------------------------------
    print("Creating node symbols")
    node_symbols = build_symbol_list(board_size)
    print(f"Total node symbols: {len(node_symbols)}")

    print("Using edge symbols from config:", config.edge.symbols)
    # We assume config.edge.symbols = ["1", "2", "Corner", "Left Side", "Right Side", "Bottom Side", "Top Side"]

    # Map cell values to string tags for node properties
    cell_value_mapping = {
        0: "Empty",
        1: "Red",   # player1
        2: "Blue",  # player2
    }

    # Edge labels for same-color neighbor connections
    edge_type_same = {
        1: "1",  # player1-player1
        2: "2",  # player2-player2
    }

    # Default neighbor edge (neutral)
    edge_type_default = "Corner"

    # Side labels (must match config.edge.symbols)
    LEFT_SIDE = "Left Side"
    RIGHT_SIDE = "Right Side"
    TOP_SIDE = "Top Side"
    BOTTOM_SIDE = "Bottom Side"

    # ------------------------------------------------------------
    # 4. Convert boards to dictionaries for easy lookup
    # ------------------------------------------------------------
    print("Converting boards to game dictionaries...")
    train_games = boards_to_games_dict(X_train, board_size)
    test_games = boards_to_games_dict(X_test, board_size)

    # ------------------------------------------------------------
    # 5. Create training graphs
    # ------------------------------------------------------------
    print("Creating training graphs")

    graphs_train = Graphs(
        train_graph_length,
        symbols=node_symbols,
        hypervector_size=config.vector.hv_size,
        hypervector_bits=config.vector.hv_bits,
    )

    for graph_id in range(train_graph_length):
        graphs_train.set_number_of_graph_nodes(graph_id, board_size ** 2)

    graphs_train.prepare_node_configuration()

    for graph_id in range(train_graph_length):
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                node = (i, j)

                # Base degree = neighbors in the Hex grid
                degree = len(edges[node])

                # Extra edges for sides (self-loops)
                side_degree = 0
                if j == 0:
                    side_degree += 1          # Left Side
                if j == board_size - 1:
                    side_degree += 1          # Right Side
                if i == 0:
                    side_degree += 1          # Top Side
                if i == board_size - 1:
                    side_degree += 1          # Bottom Side

                graphs_train.add_graph_node(
                    graph_id,
                    node_id,
                    degree + side_degree
                )


    graphs_train.prepare_edge_configuration()

    print("Adding training edges...")
    for graph_id in range(train_graph_length):
        game = train_games[graph_id]
        if graph_id % 500 == 0:
            print(f"  Adding edges for training graph id: {graph_id}")

        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                node = (i, j)
                cell_value = game.get(node_id, 0)

                # 1) Normal neighbor edges with color info
                for neighbor in edges[node]:
                    ni, nj = neighbor
                    neighbor_id = ni * board_size + nj
                    cell_neighbor = game.get(neighbor_id, 0)

                    if cell_value == cell_neighbor and cell_value != 0:
                        edge_label = edge_type_same[cell_value]
                    else:
                        edge_label = edge_type_default

                    graphs_train.add_graph_node_edge(
                        graph_id, node_id, neighbor_id, edge_label
                    )

                # 2) Side self-loop edges: Left/Right/Top/Bottom
                #    This encodes that this node touches a particular side of the board.
                if j == 0:
                    graphs_train.add_graph_node_edge(
                        graph_id, node_id, node_id, LEFT_SIDE
                    )
                if j == board_size - 1:
                    graphs_train.add_graph_node_edge(
                        graph_id, node_id, node_id, RIGHT_SIDE
                    )
                if i == 0:
                    graphs_train.add_graph_node_edge(
                        graph_id, node_id, node_id, TOP_SIDE
                    )
                if i == board_size - 1:
                    graphs_train.add_graph_node_edge(
                        graph_id, node_id, node_id, BOTTOM_SIDE
                    )

    print("Adding training node properties...")
    for graph_id in range(train_graph_length):
        game = train_games[graph_id]
        if graph_id % 500 == 0:
            print(f"  Adding properties for training graph id: {graph_id}")

        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                cell_value = game.get(node_id, 0)
                cell_property = cell_value_mapping[cell_value]

                # Content
                graphs_train.add_graph_node_property(
                    graph_id, node_id, f"{cell_property}_{i}_{j}"
                )

                # Simple connectivity heuristic
                num_same = 0
                for neighbor in edges[(i, j)]:
                    ni, nj = neighbor
                    neighbor_id = ni * board_size + nj
                    cell_neighbor = game.get(neighbor_id, 0)
                    if cell_neighbor == cell_value and cell_value != 0:
                        num_same += 1

                if num_same > 1:
                    graphs_train.add_graph_node_property(
                        graph_id, node_id, f"connected_{i}_{j}"
                    )

                # Positional
                graphs_train.add_graph_node_property(
                    graph_id, node_id, f"c{i+1}_{i}_{j}"
                )
                graphs_train.add_graph_node_property(
                    graph_id, node_id, f"r{j+1}_{i}_{j}"
                )

    print("Encoding training graphs...")
    graphs_train.encode()

    # ------------------------------------------------------------
    # 6. Create test graphs (reuse encoding)
    # ------------------------------------------------------------
    print("Creating test graphs")

    graphs_test = Graphs(
        test_graph_length,
        init_with=graphs_train
    )

    for graph_id in range(test_graph_length):
        graphs_test.set_number_of_graph_nodes(graph_id, board_size ** 2)

    graphs_test.prepare_node_configuration()

    for graph_id in range(test_graph_length):
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                node = (i, j)

                degree = len(edges[node])

                side_degree = 0
                if j == 0:
                    side_degree += 1
                if j == board_size - 1:
                    side_degree += 1
                if i == 0:
                    side_degree += 1
                if i == board_size - 1:
                    side_degree += 1

                graphs_test.add_graph_node(
                    graph_id,
                    node_id,
                    degree + side_degree
                )


    graphs_test.prepare_edge_configuration()

    print("Adding test edges...")
    for graph_id in range(test_graph_length):
        game = test_games[graph_id]
        if graph_id % 500 == 0:
            print(f"  Adding edges for test graph id: {graph_id}")

        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                node = (i, j)
                cell_value = game.get(node_id, 0)

                # Normal neighbor edges
                for neighbor in edges[node]:
                    ni, nj = neighbor
                    neighbor_id = ni * board_size + nj
                    cell_neighbor = game.get(neighbor_id, 0)

                    if cell_value == cell_neighbor and cell_value != 0:
                        edge_label = edge_type_same[cell_value]
                    else:
                        edge_label = edge_type_default

                    graphs_test.add_graph_node_edge(
                        graph_id, node_id, neighbor_id, edge_label
                    )

                # Side self-loop edges
                if j == 0:
                    graphs_test.add_graph_node_edge(
                        graph_id, node_id, node_id, LEFT_SIDE
                    )
                if j == board_size - 1:
                    graphs_test.add_graph_node_edge(
                        graph_id, node_id, node_id, RIGHT_SIDE
                    )
                if i == 0:
                    graphs_test.add_graph_node_edge(
                        graph_id, node_id, node_id, TOP_SIDE
                    )
                if i == board_size - 1:
                    graphs_test.add_graph_node_edge(
                        graph_id, node_id, node_id, BOTTOM_SIDE
                    )

    print("Adding test node properties...")
    for graph_id in range(test_graph_length):
        game = test_games[graph_id]
        if graph_id % 500 == 0:
            print(f"  Adding properties for test graph id: {graph_id}")

        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                cell_value = game.get(node_id, 0)
                cell_property = cell_value_mapping[cell_value]

                graphs_test.add_graph_node_property(
                    graph_id, node_id, f"{cell_property}_{i}_{j}"
                )

                num_same = 0
                for neighbor in edges[(i, j)]:
                    ni, nj = neighbor
                    neighbor_id = ni * board_size + nj
                    cell_neighbor = game.get(neighbor_id, 0)
                    if cell_neighbor == cell_value and cell_value != 0:
                        num_same += 1

                if num_same > 1:
                    graphs_test.add_graph_node_property(
                        graph_id, node_id, f"connected_{i}_{j}"
                    )

                graphs_test.add_graph_node_property(
                    graph_id, node_id, f"c{i+1}_{i}_{j}"
                )
                graphs_test.add_graph_node_property(
                    graph_id, node_id, f"r{j+1}_{i}_{j}"
                )

    print("Encoding test graphs...")
    graphs_test.encode()

    # ------------------------------------------------------------
    # 7. Train MultiClassGraphTsetlinMachine
    # ------------------------------------------------------------
    print("Initializing MultiClassGraphTsetlinMachine...")

    tm = MultiClassGraphTsetlinMachine(
        config.model.number_of_clauses,
        config.model.T,
        config.model.s,
        depth=config.model.depth,
        message_size=config.vector.msg_size,
        message_bits=config.vector.msg_bits,
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    )

    train_acc = []
    test_acc = []
    epoch_list = []
    epoch = 0

    print("Starting training...")
    for i in range(config.model.epochs):
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
