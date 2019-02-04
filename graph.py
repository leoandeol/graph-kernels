#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from random import random
#rajouter labels
#melanger types graphes dans bd ?
#pour n=1000, generer 5 types, 10 examples de chaque de taille differente et les alterer 20 fois ?
#rajouter poids ou non dans graphe ?
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
	
def gen_database(n, nb_colours):
	""" Generates a database of graphs
	Parameters
	----------
	n : int
		size of database
	nb_colours : int
		number of possible values per colour
	"""
	db_A = []
	for i in range(n):
		#first test
		if random()<0.5:
			G = gen_graph("ring", int(random()*5+10))
			#alter_graph(G,2)
			A_ = nx.adjacency_matrix(G).T.A
			D = np.diag(1/np.sum(A_,0))
			A = A_ @ D
			db_A.append((A,"ring"))
		else:
			G = gen_graph("star", int(random()*5+10))
			#alter_graph(G,2)
			A_ = nx.adjacency_matrix(G).T.A
			D = np.diag(1/np.sum(A_,0))
			A = A_ @ D
			db_A.append((A,"star"))
	return np.array(db_A)