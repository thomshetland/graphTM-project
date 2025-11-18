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
    """
    Correct and stable hex adjacency builder with 4 virtual side nodes.
    Coordinates:
        left   -> (-2, -1)
        right  -> (-1, -2)
        top    -> (-1, -1)
        bottom -> (-2, -2)
    Returns:
        edges[(i,j)] = list of neighbors (both real and virtual)
        edges[(virtual)] = list of real neighbors
    """
    edges = {}

    # ------------------------
    # 1. Initialize board nodes
    # ------------------------
    for i in range(board_size):
        for j in range(board_size):
            edges[(i, j)] = []

    # ------------------------
    # 2. Initialize virtual nodes
    # ------------------------
    LEFT   = (-2, -1)
    RIGHT  = (-1, -2)
    TOP    = (-1, -1)
    BOTTOM = (-2, -2)

    edges[LEFT] = []
    edges[RIGHT] = []
    edges[TOP] = []
    edges[BOTTOM] = []

    # ------------------------
    # 3. Helper for safe addition
    # ------------------------
    def connect(a, b):
        if b not in edges[a]:
            edges[a].append(b)

    # ------------------------
    # 4. Real hex adjacency
    # ------------------------
    for i in range(board_size):
        for j in range(board_size):
            node = (i, j)

            # (i+1, j)
            if i + 1 < board_size:
                connect(node, (i+1, j))
                connect((i+1, j), node)

            # (i+1, j-1)
            if i + 1 < board_size and j - 1 >= 0:
                connect(node, (i+1, j-1))
                connect((i+1, j-1), node)

            # (i, j+1)
            if j + 1 < board_size:
                connect(node, (i, j+1))
                connect((i, j+1), node)

            # (i-1, j)
            if i - 1 >= 0:
                connect(node, (i-1, j))
                connect((i-1, j), node)

            # (i, j-1)
            if j - 1 >= 0:
                connect(node, (i, j-1))
                connect((i, j-1), node)
            
            # ------------------------
            # 5. Add virtual node adjacency
            # ------------------------
            if j == 0:
                connect(node, LEFT)
                connect(LEFT, node)

            if j == board_size - 1:
                connect(node, RIGHT)
                connect(RIGHT, node)

            if i == 0:
                connect(node, TOP)
                connect(TOP, node)

            if i == board_size - 1:
                connect(node, BOTTOM)
                connect(BOTTOM, node)

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
                f"Connected_{i}_{j}",
                f"c{i+1}_{i}_{j}",
                f"r{j+1}_{i}_{j}",
            ])
    return symbols

def test_build_symbol_list(board_size):
    """Generate only the symbols that will actually be used"""
    symbols = []
    
    # Edge type symbols (these are already in config.edge.symbols)
    symbols.extend(['Plain', '1', '2'])
    
    # Cell state symbols (no position suffix!)
    symbols.extend(['Empty', 'Player1', 'Player2'])
    
    # Position symbols
    for i in range(board_size):
        symbols.append(f'col_{i}')
        symbols.append(f'row_{i}')
    
    # Connectivity symbol
    symbols.append('Connected')
    
    # Optional: degree symbols (number of same-color neighbors)
    for d in range(7):  # 0-6 neighbors on hex board
        symbols.append(f'Degree_{d}')
    
    return symbols
