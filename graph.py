#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
import scipy.sparse as sp
from random import random
import pickle as pk
from time import time


class Database:

    def __init__(self, path = None):
        # on se concentre sur star, ring et tree
        self.GRAPH_TYPES = ["ring", "star", "tree"]
        if path is None:
            self.loaded = False
            self.path = None
        else:
            self.loaded = True
            self.db = self.import_db(path)
            self.path = path

    def gen_graph(self, type, n, nb_colors):
        G = None
        if type == "ring":
            G = nx.cycle_graph(n)
        elif type == "star":
            G = nx.star_graph(n)
        elif type == "tree":
            G = nx.balanced_tree(2,int(np.floor(np.log2(n)-1)))
            while len(G.nodes()) < n:
                node = np.random.choice(G.nodes())
                if G.degree[node]==1:
                    node2 = len(G.nodes())
                    G.add_node(node2)
                    G.add_edge(node,node2)
        # elif type == "grid":
        #     G = nx.grid_2d_graph(n//2,n//2)
        # elif type == "cube":
        #     G = nx.hypercube_graph(int(np.floor(np.log2(n))))
        # elif type == "chain":
        #     G = nx.path_graph(n)
        else:
            return "Error"
        for (u,v) in G.edges():
            G.edges[u,v]["label"] = np.random.randint(nb_colors)
        return G
    
    def alter_graph_struct(self, G_orig, type, n):
        G = G_orig.copy()
        if type == "star":
            if random()<0.5:
                G = nx.star_graph(len(G.nodes())-n)
            else:
                G = nx.star_graph(len(G.nodes())+n)
        elif type == "ring": 
            if random()<0.5:
                G = nx.cycle_graph(len(G.nodes())-n)
            else:
                G = nx.cycle_graph(len(G.nodes())+n)
        elif type == "tree":
            if random()<0.5:
                while n > 0:
                    no = np.random.choice(G.nodes())
                    if G.degree[no]==1:
                        n -= 1
                        G.remove_nodes_from([no])
            else:
                while n > 0:
                    no = np.random.choice(G.nodes())
                    if G.degree[no]==1:
                        n -= 1
                        no2 = len(G.nodes())
                        G.add_node(no2)
                        G.add_edge(no,no2)
        else:
            raise NotImplementedError
        return G
            
    def alter_graph_labels(self, G, n):
        for (u,v) in map(tuple,np.random.permutation(G.edges())[:n]):
            G.edges[u,v]["label"]=np.random.randint(n)
            
        
    def gen_and_draw(self, type, n, quantif):
        """
        quantif : int
        number of values the label can take
        """
        nx.draw(self.gen_graph(type,n,quantif))
        plt.show()
        
    def product_graph(self, X,Y):
        A = nx.adjacency_matrix(X)
        B = nx.adjacency_matrix(Y)
        W = sp.kron(A,B)
        G = nx.from_scipy_sparse_matrix(W)
        return G, W

    def gen_database(self, nb_graphs, nb_altered, nb_nodes, nb_colors, intensity, normalized=False):
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
                if normalized:
                    D = np.diagflat(1/np.sum(A_,axis=0))
                    A = A_ @ D
                else:
                    A = A_
                    db_A.append((A,typ))
                    #np.random.shuffle(db_A)
        return np.array(db_A)

        def export_db(self, db, path):
            if not self.loaded:
                return False
            self.path = path
            with open(path,"wb") as f:
                pk.dump(db,f)
                
        def import_db(self, path):
            self.loaded = true
            self.path = path
            with open(path,"rb") as f:
                return pk.load(f)

        def gen_export_db(self, nb_graphs, nb_altered, nb_nodes, nb_colors, intensity, normalized=True):
            db = gen_database(nb_graphs, nb_altered, nb_nodes, nb_colors, intensity, normalized)
            path = str("dbs/db-"+str(time())+".dat")
            self.export_db(db,path)
            self.loaded = True
            self.path = path
            return (db,path)
