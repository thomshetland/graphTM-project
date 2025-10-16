from GraphTsetlinMachine.graphs import Graphs
import numpy as np
import numba
import random

def main():
    print("Hello world")

if __name__ == "__main__":
    graphs_train = Graphs(
        10000, # sets how many graphs to make
        symbols = ['A', 'B'], # symbols for each feature (XOR has 2 feature)
        hypervector_size = 32, # how large the hypervector is to store symbols.
        hypervector_bits = 2, # set how many bits to use to represent symbols
    )

    for graph_id in range(10000):
        graphs_train.set_number_of_graph_nodes(graph_id, 2)

    graphs_train.prepare_node_configuration()

    for graph_id in range(10000):
        number_of_outgoing_edges = 1
        graphs_train.add_graph_node(graph_id, 'Node 1', number_of_outgoing_edges)
        graphs_train.add_graph_node(graph_id, 'Node 2', number_of_outgoing_edges)

    graphs_train.prepare_edge_configuration()

    for graph_id in range(10000):
        edge_type = "Plain"
        graphs_train.add_graph_node(graph_id, 'Node 1', 'Node 2', edge_type) 
        graphs_train.add_graph_node(graph_id, 'Node 2', 'Node 1', edge_type)
        # two edges for both directions inbetween the nodes

    Y_train = np.empty(10000, dtype=np.uint32)
    for graph_id in range(10000):
        x1 = random.choice(['A', 'B'])
        x2 = random.choice(['A', 'B'])

        graphs_train.add_graph_node_property(graph_id, 'Node 1', x1)
        graphs_train.add_graph_node_property(graph_id, 'Node 2', x2)

    if x1 == x2:
        Y_train[graph_id] = 0
    else: 
        Y_train[graph_id] = 1

    if np.random.rand() <= 0.01:
        Y_train[graph_id] = 1 - Y_train[graph_id]
        