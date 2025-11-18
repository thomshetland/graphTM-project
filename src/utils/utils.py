import numpy as np
from typing import List
from config import config

def transform_dataset(flat_dataset: List[List[int]]):
    new_dataset = []
    for flat_board in flat_dataset:
        board = np.array(flat_board).reshape(config.game.board_size, config.game.board_size)
        new_dataset.append(board)
    return np.asarray(new_dataset)

def boards_to_games_dict(X, board_size):
    """
    X: array-like (n_samples, board_size, board_size) or (n_samples, board_size*board_size)
    Returns: list[dict[node_id -> cell_value]]
    cell_value: 0 (empty), 1 (player1), 2 (player2)
    """
    X_arr = np.asarray(X)
    games = []
    num_nodes = board_size * board_size
    for board in X_arr:
        board_flat = board.reshape(-1)
        game_state = {idx: int(val) for idx, val in enumerate(board_flat)}
        game_state[num_nodes] = 0
        game_state[num_nodes + 1] = 0
        game_state[num_nodes + 2] = 0
        game_state[num_nodes + 3] = 0
        games.append(game_state)
    return games

def build_hex_adjacency(board_size):
    edges = {}
    for i in range(board_size):
        for j in range(board_size):
            node = (i, j)
            edges[node] = []

    left_node = (-2, -1)
    right_node = (-1, -2)
    top_node = (-1, -1)
    bottom_node = (-2, -2)

    edges[left_node] = []
    edges[right_node] = []
    edges[top_node] = []
    edges[bottom_node] = []

    def connectNodes(a, b):
        if b not in edges[a]:
            edges[a].append(b)

    for i in range(board_size):
        for j in range(board_size):
            node = (i, j)

            if i < board_size - 1:
                connectNodes(node, (i + 1, j))
                connectNodes((i + 1, j), node)

                if j > 0:
                    connectNodes(node, (i + 1, j - 1))
                    connectNodes((i + 1, j - 1), node)

            if j < board_size - 1:
                connectNodes(node, (i, j + 1))
                connectNodes((i, j + 1), node)

            if i > 0:
                connectNodes(node, (i - 1, j))
                connectNodes((i - 1, j), node)

            if j > 0:
                connectNodes(node, (i, j - 1))
                connectNodes((i, j - 1), node)
            
            # Left border
            if j == 0:
                connectNodes(node, left_node)
                connectNodes(left_node, node)

            # Right border
            if j == board_size - 1:
                connectNodes(node, right_node)
                connectNodes(right_node, node)

            # Top border
            if i == 0:
                connectNodes(node, top_node)
                connectNodes(top_node, node)

            # Bottom border
            if i == board_size - 1:
                connectNodes(node, bottom_node)
                connectNodes(bottom_node, node)
    return edges


def build_symbol_list(board_size):
    """Our node symbol approach."""
    symbols = []
    for i in range(board_size):
        for j in range(board_size):
            symbols.extend([
                f"Empty_{i}_{j}",
                f"Player1_{i}_{j}",
                f"Player2_{i}_{j}",
                f"connected_{i}_{j}",
                f"c{i+1}_{i}_{j}",
                f"r{j+1}_{i}_{j}",
            ])
    return symbols

