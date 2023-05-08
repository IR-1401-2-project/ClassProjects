# We want to implement the pagerank algorithm
# We want to implement the iterative algorithm
# We use the .dat files in the resources folder

base_path = '/home/toorajtaraz/Documents/university/IR/ClassProjects/PageRank/'


# We define the function that calculates the pagerank and outputs each iteration
def output_dat(res, output_dat):
    # we will save each float in res in a separate line in output_dat
    with open(output_dat, 'w') as f:
        for i in range(len(res)):
            f.write(str(res[i]) + '\n')

class Node:
    def __init__(self, name):
        self.name = name
        self.outgoing = []
        self.incoming = []
        self.rank = 1.0
    def add_outgoing(self, node):
        self.outgoing.append(node)
    def add_incoming(self, node):
        self.incoming.append(node)
    def get_rank(self):
        return self.rank
    def update_rank(self, number_of_nodes, damping_factor=0.1):
        incoming = self.incoming
        pagerank = sum((node.get_rank() / len(node.outgoing)) for node in incoming)
        random_jumping = damping_factor / number_of_nodes
        self.rank = random_jumping + (1 - damping_factor) * pagerank

class Graph:
    def __init__(self):
        self.nodes = []
    def add_node(self, node):
        self.nodes.append(node)
    def get_nodes(self):
        return self.nodes
    def init_graph(self, matrix, number_of_nodes, number_of_edges):
        for i in range(number_of_nodes):
            self.add_node(Node(i))
        for i in range(number_of_nodes):
            for j in range(number_of_nodes):
                if matrix[i][j] == 1:
                    self.nodes[i].add_outgoing(self.nodes[j])
                    self.nodes[j].add_incoming(self.nodes[i])
    def get_page_rank(self, iters=10):
        for _ in range(iters):
            for node in self.nodes:
                node.update_rank(len(self.nodes))
        output_dat([node.get_rank() for node in self.nodes], base_path + f'iter{iters}.res')
        return [node.get_rank() for node in self.nodes]

import numpy as np

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
        # graph = Graph()
        # graph.init_graph(matrix, nodes_num, edges_num)
        # del matrix  
        return matrix, nodes_num, edges_num
    
test_graph = read_dat_file('/home/toorajtaraz/Downloads/IR_lab9_examples/examples/smallrmat.dat')
print(test_graph)
# print(test_graph[0].get_page_rank(24))

from scipy.sparse import csr_matrix

def mem_eff_pagerank(_matrix, damping_factor=0.1, max_iterations=100, tolerance=1e-6):
    matrix = csr_matrix(_matrix)
    del _matrix
    row_sum = np.array(matrix.sum(axis=1)).flatten()
    row_sum[row_sum != 0] = 1 / row_sum[row_sum != 0]
    matrix = matrix.multiply(row_sum[:, np.newaxis])
    
    # Initialize the page rank vector
    n = matrix.shape[0]
    rank = {i: 1.0 / n for i in range(n)}
    
    # Initialize the incoming and outgoing edges for each node
    incoming = {i: [] for i in range(n)}
    outgoing = {i: [] for i in range(n)}
    for i, j in np.argwhere(matrix):
        incoming[j].append(i)
        outgoing[i].append(j)
    
    # Iterate until convergence or maximum number of iterations
    for iteration in range(max_iterations):
        prev_rank = rank.copy()
        for i in range(n):
            pagerank = sum(prev_rank[j] / len(outgoing[j]) for j in incoming[i])
            rank[i] = damping_factor / n + (1 - damping_factor) * pagerank
        if sum(abs(rank[i] - prev_rank[i]) for i in range(n)) < tolerance:
            break
    
    return [rank[i] for i in range(n)]

res = mem_eff_pagerank(test_graph[0], 0.1, 10, 1e-6)
print(res)
output_dat(res, base_path + 'iter10.res')