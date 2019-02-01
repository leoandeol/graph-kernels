#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

def gen_graph(type, n):
    if type == "ring":
        return nx.cycle_graph(n)
    elif type == "star":
        return nx.star_graph(n)
    elif type == "grid":
        return nx.grid_graph(n)
    elif type == "tree":
        pass
    elif type == "cube":
        return nx.hypercube_graph(d)
    elif type == "chain":
        pass
    elif type == "ladder":
        return nx.ladder_graph(n//2)
    else:
        return "Error"

#alter nodes or edges or labels ?
def alter_graph(G, nodes):
    G.remove_nodes(np.random.choice(G.nodes,nodes))
    
def gen_and_draw(type, n):
    nx.draw(gen_graph(type,n))
    plt.show()

def product_graph(X,Y):
	A = nx.adjacency_matrix(X)
	B = nx.adjacency_matrix(Y)
	W = sp.kron(A,B)
	G = nx.from_scipy_sparse_matrix(W)
	return G, W