from GraphTsetlinMachine.graphs import Graphs
import numpy as np
import numba

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