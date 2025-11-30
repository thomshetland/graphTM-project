import numpy as np
from typing import List, Optional
from config import config
from GraphTsetlinMachine.graphs import Graphs

def transform_dataset(flat_dataset: List[List[int]]):
    new_dataset = []
    for flat_board in flat_dataset:
        board = np.array(flat_board).reshape(config.game.board_size, config.game.board_size)
        new_dataset.append(board)
    return np.asarray(new_dataset)

def boards_to_games_dict(X, board_size):
    X_arr = np.asarray(X)
    games = []
    num_nodes = board_size * board_size
    for board in X_arr:
        board_flat = board.reshape(-1)
        game_state = {idx: int(val) for idx, val in enumerate(board_flat)}
        game_state[num_nodes] = 2
        game_state[num_nodes + 1] = 2
        game_state[num_nodes + 2] = 1
        game_state[num_nodes + 3] = 1
        games.append(game_state)
    return games

def build_hex_adjacency(board_size):
    edges = {}
    for i in range(board_size):
        for j in range(board_size):
            edges[(i, j)] = []

    LEFT   = (-2, -1)
    RIGHT  = (-1, -2)
    TOP    = (-1, -1)
    BOTTOM = (-2, -2)

    edges[LEFT] = []
    edges[RIGHT] = []
    edges[TOP] = []
    edges[BOTTOM] = []

    def connect(a, b):
        if b not in edges[a]:
            edges[a].append(b)

    for i in range(board_size):
        for j in range(board_size):
            node = (i, j)

            if i + 1 < board_size:
                connect(node, (i+1, j))
                connect((i+1, j), node)

            if i + 1 < board_size and j - 1 >= 0:
                connect(node, (i+1, j-1))
                connect((i+1, j-1), node)

            if j + 1 < board_size:
                connect(node, (i, j+1))
                connect((i, j+1), node)

            if i - 1 >= 0:
                connect(node, (i-1, j))
                connect((i-1, j), node)

            if j - 1 >= 0:
                connect(node, (i, j-1))
                connect((i, j-1), node)
            
            if i - 1 >= 0 and j + 1 < board_size:
                connect(node, (i-1, j+1))
                connect((i-1, j+1), node)
            
            # virtual nodes
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
    symbols = set()
    for i in range(board_size):
        for j in range(board_size):
            symbols.update([
                f"Empty",
                f"Player1",
                f"Player2",
                f"Placement_{i}_{j}",
                f"Connected_{i}_{j}",
                f"c{i}",
                f"r{j}",
            ])
    symbols = list(symbols)
    return symbols

def build_graphs(
    ds_type,
    graphs_train: Optional[Graphs] = None,
    X=None,
    games=None,
    symbols=None,
    edges=None,
    board_size=None,
    hv_size=None,
    hv_bits=None
):
    hv_size = len(symbols)
    print(f"{ds_type} has {hv_size} symbols")
    n_board = board_size**2
    graph_length = X.shape[0]
    if ds_type == "Train":
        graphs = Graphs(
            graph_length,
            symbols=symbols,
            hypervector_size=hv_size,
            hypervector_bits=hv_bits,
            double_hashing=True,
            one_hot_encoding=True
        )
    else:
        graphs = Graphs(
            graph_length,
            init_with=graphs_train
        )
    
    for graph_id in range(graph_length):
        graphs.set_number_of_graph_nodes(graph_id, (board_size**2) + 4)

    print(f"Preparing node configuration for {ds_type}")
    graphs.prepare_node_configuration()

    for graph_id in range(graph_length):
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                node = (i, j)

                outgoing_edges = len(edges[node])

                graphs.add_graph_node(
                    graph_id,
                    node_id,
                    outgoing_edges
                )
        for i in range(4):
            node_id = n_board + i
            graphs.add_graph_node(
                graph_id,
                node_id,
                board_size
            )

    print(f"Preparing edge configuration for {ds_type}")
    graphs.prepare_edge_configuration()
    edge_type_mapping = {
        0: 'Plain',
        1: 'Player 1',
        2: 'Player 2'
    }
    for graph_id in range(graph_length):
        game = games[graph_id]
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                node = (i, j)      
                cell_value = game.get(node_id, 0)
                for neighbor in edges[node]:
                    ni, nj = neighbor
                    if ni < 0:
                        if ni == -2 and nj == -1: # left
                            neighbor_id = n_board 
                            cell_neighbor = 2
                        elif ni == -1 and nj == -2: # right
                            neighbor_id = n_board + 1
                            cell_neighbor = 2
                        elif ni == -1 and nj == -1: # top
                            neighbor_id = n_board + 2 
                            cell_neighbor = 1
                        else:                       # bottom
                            neighbor_id = n_board + 3
                            cell_neighbor = 1
                        
                        if cell_value == cell_neighbor and cell_value != 0:
                            edge_label = edge_type_mapping[cell_value]
                        else:
                            edge_label = edge_type_mapping[0]
                        graphs.add_graph_node_edge(
                            graph_id,
                            node_id,
                            neighbor_id,
                            edge_label
                        )
                        graphs.add_graph_node_edge(
                            graph_id,
                            neighbor_id,
                            node_id,
                            edge_label
                        )
                    else:
                        neighbor_id = ni * board_size + nj
                        cell_neighbor = game.get(neighbor_id, 0)
                        if cell_value == cell_neighbor and cell_value != 0:
                            edge_label = edge_type_mapping[cell_value] 
                        else:
                            edge_label = edge_type_mapping[0]
                        
                        graphs.add_graph_node_edge(
                            graph_id,
                            node_id,
                            neighbor_id,
                            edge_label
                        )

    print(f"Adding {ds_type} node properties")
    cell_value_mapping = {
        0: "Empty",
        1: "Player1",   
        2: "Player2",  
    }
    for graph_id in range(graph_length):
        game = games[graph_id]
        for i in range(board_size):
            for j in range(board_size):
                node_id = i * board_size + j
                cell_value = game.get(node_id, 0)
                cell_property = cell_value_mapping[cell_value]
                graphs.add_graph_node_property(
                    graph_id, node_id, f"{cell_property}"
                )

                graphs.add_graph_node_property(
                    graph_id, node_id, f"c{i}"
                )
                graphs.add_graph_node_property(
                    graph_id, node_id, f"r{j}"
                )

                graphs.add_graph_node_property(
                    graph_id, node_id, f"Placement_{i}_{j}"
                )
                
                num_same = 0
                node = (i, j)
                for neighbor in edges[node]:
                    ni, nj = neighbor
                    if ni < 0:
                        if ni == -2 and nj == -1:
                            neighbor_id = n_board
                        elif ni == -1 and nj == -2:
                            neighbor_id = n_board + 1
                        elif ni == -1 and nj == -1:
                            neighbor_id = n_board + 2
                        else:
                            neighbor_id = n_board + 3
                    else:
                        neighbor_id = ni * board_size + nj

                    cell_neighbor = game.get(neighbor_id, 0)
                    if cell_neighbor == cell_value and cell_value != 0:
                        num_same += 1
                
                if num_same > 1:
                    graphs.add_graph_node_property(
                        graph_id, node_id, f"Connected_{i}_{j}"
                    )
    graphs.encode()
    return graphs