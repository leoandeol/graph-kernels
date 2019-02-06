#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import scipy.sparse as sp
from random import random
import pickle as pk
from time import time


GRAPH_TYPES = ["ring", "star", "grid", "tree", "cube", "chain"]

def gen_graph(type, n, nb_colors):
    G = None
    if type == "ring":
        G = nx.cycle_graph(n)
    elif type == "star":
        G = nx.star_graph(n)
    elif type == "grid":
        G = nx.grid_2d_graph(n//2,n//2)
    elif type == "tree":
        G = nx.balanced_tree(2,int(np.floor(np.log2(n))))
    elif type == "cube":
        G = nx.hypercube_graph(int(np.floor(np.log2(n))))
    elif type == "chain":
        G = nx.path_graph(n)
    else:
        return "Error"
    for (u,v) in G.edges():
        G.edges[u,v]["label"] = np.random.randint(nb_colors)
    return G

def alter_graph_nodes(G, n):
    G.remove_nodes_from(map(tuple,np.random.permutation(G.nodes())[:n]))
                        
def alter_graph_edges(G, n):
    G.remove_edges_from(map(tuple,np.random.permutation(G.edges())[:n]))

def alter_graph_labels(G, n):
    for (u,v) in map(tuple,np.random.permutation(G.edges())[:n]):
        G.edges[u,v]["label"]=np.random.randint(n)
        
    
def gen_and_draw(type, n, quantif):
    """
    quantif : int
    number of values the label can take
    """
    nx.draw(gen_graph(type,n))
    plt.show()

def product_graph(X,Y):
    A = nx.adjacency_matrix(X)
    B = nx.adjacency_matrix(Y)
    W = sp.kron(A,B)
    G = nx.from_scipy_sparse_matrix(W)
    return G, W
    
def gen_database(nb_graphs, nb_altered, nb_nodes, nb_colors, intensity):
    """ Generates a database of graphs
	Parameters
	----------
	nb_graphs : int
		number of random graphs
        nb_altered : int
                number of altered versions of a graph
        nb_nodes : int
                number of nodes per graph
	nb_colours : int
		number of possible values per colour
    intensity : float
          ]0;1[ intensity of alteration
	"""
    db_A = []
    for i in range(nb_graphs):
        #source graph
        typ = np.random.choice(GRAPH_TYPES)
        GS = gen_graph(typ,nb_nodes,nb_colors)
        if GS == "Error":
            print("Error")
        A_ = nx.to_numpy_matrix(GS).T
        D = np.diagflat(1/np.sum(A_,axis=0))
        A = A_ @ D
        db_A.append((A,typ))
        for _ in range(nb_altered):
            G = GS.copy()
            alter_graph_nodes(GS,np.random.randint(max(1,int(np.floor(nb_nodes*intensity)))))
            alter_graph_edges(GS,np.random.randint(max(1,int(np.floor(nb_nodes*intensity)))))
            alter_graph_labels(GS,np.random.randint(max(1,int(np.floor(nb_nodes*intensity)))))
            A_ = nx.to_numpy_matrix(G).T
            D = np.diagflat(1/np.sum(A_,axis=0))
            A = A_ @ D
            db_A.append((A,typ))
    np.radom.shuffle(db_A)
    return np.array(db_A)
        
def export_db(db,path):
    with open(path,"wb") as f:
        pk.dump(db,f)
                
def import_db(path):
    with open(path,"rb") as f:
        return pk.load(f)


def gen_export_db(nb_graphs, nb_altered, nb_nodes, nb_colors, intensity):
    db = gen_database(nb_graphs, nb_altered, nb_nodes, nb_colors, intensity)
    path = str("dbs/db-"+str(time())+".dat")
    export_db(db,path)
    return (db,path)
