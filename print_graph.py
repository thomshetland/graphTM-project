#!/usr/bin/env python3
# preview_graphs_train.py
# Build GraphTM graphs for Hex and print a readable preview (no training)

from GraphTsetlinMachine.graphs import Graphs
# from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine  # not needed for preview

from utils.x_builder import build_boards_from_moves

import numpy as np
import argparse
from sklearn.model_selection import train_test_split


def make_hex_adjacency(board_size: int):
    """Return a dict: (i,j) -> list[(ni,nj)] with 6-neighbor Hex adjacency."""
    edges = {(i, j): [] for i in range(board_size) for j in range(board_size)}

    def add(a, b):
        if b not in edges[a]:
            edges[a].append(b)

    for i in range(board_size):
        for j in range(board_size):
            node = (i, j)
            # Down
            if i < board_size - 1:
                add(node, (i + 1, j))
                add((i + 1, j), node)
                # Down-left (Hex diagonal)
                if j > 0:
                    add(node, (i + 1, j - 1))
                    add((i + 1, j - 1), node)
            # Right
            if j < board_size - 1:
                add(node, (i, j + 1))
                add((i, j + 1), node)
            # Up
            if i > 0:
                add(node, (i - 1, j))
                add((i - 1, j), node)
            # Left
            if j > 0:
                add(node, (i, j - 1))
                add((i, j - 1), node)

    return edges


def to_game_states(X_flat: np.ndarray):
    """
    Convert X with shape (N, board_size*board_size) into a list[dict[node_id]->cell_value].
    cell_value must be ints {0: Empty, 1: Red, 2: Blue}.
    """
    games = []
    for row in X_flat:
        gs = {int(node_id): int(val) for node_id, val in enumerate(row)}
        games.append(gs)
    return games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/hex_5x5_5000.npz", type=str)
    parser.add_argument("--board-size", default=5, type=int)

    # GraphTM encoding knobs (we’re just previewing; these are sane defaults)
    parser.add_argument("--hypervector-size", default=1024, type=int)
    parser.add_argument("--hypervector-bits", default=16, type=int)
    parser.add_argument("--message-size", default=512, type=int)
    parser.add_argument("--message-bits", default=16, type=int)
    parser.add_argument("--double-hashing", dest="double_hashing", action="store_true", default=False)

    # Preview options
    parser.add_argument("--show-graphs", default=3, type=int, help="How many graphs to print")
    parser.add_argument("--show-nodes", default=6, type=int, help="How many nodes per graph to print")
    args = parser.parse_args()

    print("Loading dataset…")
    data = np.load(args.dataset)
    moves = data["moves"]
    lengths = data["lengths"]
    y_ds = data["winners"]  # labels (unused in preview)

    # Build board arrays from move sequences (provided by your project)
    x_ds = build_boards_from_moves(moves, lengths, offset=0)
    # Expect x_ds shape = (N, board_size*board_size) with values in {0,1,2}
    print("Final X shape:", x_ds.shape)

    print("Splitting train/test 80/20…")
    X_train, X_test, y_train, y_test = train_test_split(x_ds, y_ds, test_size=0.2, random_state=42)
    train_graph_length = len(X_train)
    print(f"Training examples: {train_graph_length} | Test examples: {len(X_test)}")

    board_size = args.board_size
    num_nodes = board_size * board_size

    print("Creating Hex adjacency (nodes and edges)…")
    edges = make_hex_adjacency(board_size)

    # Mapping values to human-friendly names
    cell_value_mapping = {0: "Empty", 1: "Red", 2: "Blue"}
    edge_type_mapping = {0: "plain", 1: "Red", 2: "Blue"}

    # ---- SYMBOLS -----------------------------------------------------------
    # Per-cell occupancy + optional structure/positional tags
    # We’ll assign symbols at the (i,j) level to keep the encoding localized.
    print("Building symbol vocabulary…")
    symbols = []
    for i in range(board_size):
        for j in range(board_size):
            symbols.extend([
                f"Empty_{i}_{j}",
                f"Red_{i}_{j}",
                f"Blue_{i}_{j}",
                f"connected_{i}_{j}",  # local 2+ same-color neighbors
                f"c{i+1}_{i}_{j}",     # row tag, localized
                f"r{j+1}_{i}_{j}",     # col tag, localized
            ])

    # ---- GRAPHS ------------------------------------------------------------
    print("Initializing Graphs() for training set…")
    graphs_train = Graphs(
        train_graph_length,
        symbols=symbols,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        double_hashing=args.double_hashing
    )

    # Turn X_train rows into list of {node_id: cell_value}
    train_games = to_game_states(X_train)

    print("Setting node counts per graph…")
    for graph_id in range(train_graph_length):
        graphs_train.set_number_of_graph_nodes(graph_id, num_nodes)

    graphs_train.prepare_node_configuration()

    print("Adding nodes (with outgoing degree counts)…")
    # Add each node with its number of outgoing edges (based on adjacency)
    for graph_id in range(train_graph_length):
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                number_of_outgoing_edges = len(edges[(i, j)])
                graphs_train.add_graph_node(graph_id, node_id, number_of_outgoing_edges)

    graphs_train.prepare_edge_configuration()

    print("Adding edges with types (plain / Red / Blue)…")
    for graph_id in range(train_graph_length):
        game = train_games[graph_id]
        if graph_id % 1000 == 0:
            print(f"  Edges for graph {graph_id}/{train_graph_length}")
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                cell_value = game.get(node_id, 0)
                default_edge_type = edge_type_mapping[0]  # "plain"
                for (ni, nj) in edges[(i, j)]:
                    neighbor_id = ni * board_size + nj
                    cell_neighbor = game.get(neighbor_id, 0)
                    if cell_value == cell_neighbor and cell_value != 0:
                        etype = edge_type_mapping[cell_value]  # "Red" or "Blue"
                    else:
                        etype = default_edge_type
                    graphs_train.add_graph_node_edge(graph_id, node_id, neighbor_id, etype)

    print("Adding node properties (symbols per node)…")
    for graph_id in range(train_graph_length):
        game = train_games[graph_id]
        if graph_id % 1000 == 0:
            print(f"  Properties for graph {graph_id}/{train_graph_length}")
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                cell_value = game.get(node_id, 0)
                # Occupancy symbol
                graphs_train.add_graph_node_property(graph_id, node_id, f"{cell_value_mapping[cell_value]}_{i}_{j}")
                # Local same-color connectivity: 2+ neighbors of same color
                if cell_value != 0:
                    same = 0
                    for (ni, nj) in edges[(i, j)]:
                        nid = ni * board_size + nj
                        if game.get(nid, 0) == cell_value:
                            same += 1
                    if same > 1:
                        graphs_train.add_graph_node_property(graph_id, node_id, f"connected_{i}_{j}")
                # Positional tags (localized)
                graphs_train.add_graph_node_property(graph_id, node_id, f"c{i+1}_{i}_{j}")
                graphs_train.add_graph_node_property(graph_id, node_id, f"r{j+1}_{i}_{j}")

    print("Encoding graphs (hypervectors + messages)…")
    graphs_train.encode()
    print("Encoding complete.")

    # ---- PREVIEW -----------------------------------------------------------
    # The Graphs API doesn’t guarantee public getters for internals, so we’ll
    # print a clear preview using what we *do* know (our adjacency & properties).
    # Additionally, try to show hypervector stats if available.
    print("\n================ Graphs Train Preview ================\n")
    print(f"Graphs count        : {train_graph_length}")
    print(f"Board size          : {board_size} x {board_size}")
    print(f"Nodes per graph     : {num_nodes}")
    print(f"Symbols in vocab    : {len(symbols)}")
    print(f"hypervector_size    : {args.hypervector_size}")
    print(f"hypervector_bits    : {args.hypervector_bits}")
    print(f"message_size        : {args.message_size}")
    print(f"message_bits        : {args.message_bits}")
    print(f"double_hashing      : {args.double_hashing}")

    # Try to display any available internals (best effort).
    try:
        hv = getattr(graphs_train, "hypervectors", None)
        if hv is not None:
            print(f"\n[Info] graphs_train.hypervectors shape: {np.array(hv).shape}")
    except Exception as e:
        print(f"[Info] Could not access graphs_train.hypervectors: {e}")

    # Pretty print a few graphs using our known construction
    show_graphs = min(args.show_graphs, train_graph_length)
    show_nodes = min(args.show_nodes, num_nodes)

    for g in range(show_graphs):
        print(f"\n--- Graph {g} ---")
        game = train_games[g]

        # Show first few nodes' neighbors (from our adjacency dict)
        print("Neighbors (first few nodes):")
        for idx in range(show_nodes):
            i, j = divmod(idx, board_size)
            print(f"  Node {idx} {(i,j)} -> {edges[(i,j)]}")

        # Show first few nodes' properties as we added them
        print("Node properties (first few nodes):")
        for idx in range(show_nodes):
            i, j = divmod(idx, board_size)
            cell_value = game.get(idx, 0)
            props = [f"{cell_value_mapping[cell_value]}_{i}_{j}",
                     f"c{i+1}_{i}_{j}",
                     f"r{j+1}_{i}_{j}"]
            # recompute connected flag here for transparency
            if cell_value != 0:
                same = 0
                for (ni, nj) in edges[(i, j)]:
                    nid = ni * board_size + nj
                    if game.get(nid, 0) == cell_value:
                        same += 1
                if same > 1:
                    props.append(f"connected_{i}_{j}")
            print(f"  Node {idx} {(i,j)}: {props}")

    print("\n================ End of Preview ======================\n")


if __name__ == "__main__":
    main()
