# We want to implement the pagerank algorithm
# We want to implement the iterative algorithm
# We use the .dat files in the resources folder

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# We define the function that reads the file and returns the matrix
def read_dat_file(path_to_dat):
    with open(path_to_dat, 'r') as f:
        lines = f.readlines()
        # File format is:
        # a
        # b
        # x1 x2
        # x3 x4
        # ...
        # a is the number of nodes
        # b is the number of edges
        # x1 x2 is the edge from node x1 to node x2
        # We want to create a matrix of size a x a
        # We want to fill the matrix with 1 if there is an edge from x1 to x2
        # We want to fill the matrix with 0 if there is no edge from x1 to x2
        # We want to return the matrix
        nodes_num = int(lines[0])
        edges_num = int(lines[1])
        matrix = np.zeros((nodes_num, nodes_num))
        for i in range(2, len(lines)):
            line = lines[i].split()
            matrix[int(line[0])][int(line[1])] = 1
        return matrix, nodes_num, edges_num
    
test_graph = read_dat_file('/home/toorajtaraz/Downloads/IR_lab9_examples/examples/example1.dat')
base_path = '/home/toorajtaraz/Downloads/IR_lab9_examples/examples/'
print(test_graph)

# We define the function that calculates the pagerank and outputs each iteration
def output_dat(res, output_dat):
    # we will save each float in res in a separate line in output_dat
    with open(output_dat, 'w') as f:
        for i in range(len(res)):
            f.write(str(res[i]) + '\n')

def iterative_pagerank(matrix, iters=10, damping_factor=0.85, debug=False):
    # Our matrix is just a 2d array, 1s indicate edges and 0s indicate no edges
    # Damping factor is the probability that the user will continue clicking on links
    number_of_nodes = matrix.shape[0]
    # We want to create a vector of size number_of_nodes
    # We want to fill it with 1/number_of_nodes
    result_vector = np.full((number_of_nodes), 1/number_of_nodes)
    for i in range(iters):
        result_vector = (1 - damping_factor) / number_of_nodes + damping_factor * np.matmul(matrix, result_vector)
        if debug:
            print(result_vector)
            output_dat(result_vector, base_path + f'iter{i}.res')
    return result_vector

test_res = iterative_pagerank(test_graph[0], debug=True)
