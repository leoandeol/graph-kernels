#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
import scipy.sparse as sp
from random import random
import pickle as pk
from time import time
from grakel import datasets, Graph


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
                for (u,v) in G.edges():
                    #print(np.random.permutation(list(G_orig.edges(data=True)))[0][2])
                    G.edges[u,v]["label"] = np.random.permutation(list(G_orig.edges(data=True)))[0][2]["label"]
            else:
                G = nx.star_graph(len(G.nodes())+n)
                for (u,v) in G.edges():
                    #print(np.random.permutation(list(G_orig.edges(data=True)))[0][2])
                    G.edges[u,v]["label"] = np.random.permutation(list(G_orig.edges(data=True)))[0][2]["label"]
        elif type == "ring": 
            if random()<0.5:
                G = nx.cycle_graph(len(G.nodes())-n)
                for (u,v) in G.edges():
                    #print(np.random.permutation(list(G_orig.edges(data=True)))[0][2])
                    G.edges[u,v]["label"] = np.random.permutation(list(G_orig.edges(data=True)))[0][2]["label"]
            else:
                G = nx.cycle_graph(len(G.nodes())+n)
                for (u,v) in G.edges():
                    #print(np.random.permutation(list(G_orig.edges(data=True)))[0][2])
                    G.edges[u,v]["label"] = np.random.permutation(list(G_orig.edges(data=True)))[0][2]["label"]
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
                        G.add_edge(no,no2,label=np.random.permutation(list(G.edges(data=True)))[0][2]["label"])
        else:
            raise NotImplementedError
        return G
    
    def alter_graph_labels(self, G_orig, n):
        G = G_orig.copy()
        for (u,v) in map(tuple,np.random.permutation(G.edges())[:n]):
            G.edges[u,v]["label"]=np.random.randint(n)
        return G
            
        
    def gen_and_draw(self, type, n, quantif):
        """
        quantif : int
        number of values the label can take
        """
        G=self.gen_graph(type,n,quantif)
        pos=nx.spring_layout(G)
        nx.draw(G,pos)
        nx.draw_networkx_edge_labels(G,pos)
        plt.show()
        
    def product_graph(self, X,Y):
        A = nx.adjacency_matrix(X)
        B = nx.adjacency_matrix(Y)
        W = sp.kron(A,B)
        G = nx.from_scipy_sparse_matrix(W)
        return G, W

    def gen_database(self, nb_graphs, nb_altered, nb_nodes, nb_colors, intensity):
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
            typ = np.random.choice(self.GRAPH_TYPES)
            GS = self.gen_graph(typ,nb_nodes,nb_colors)
            if GS == "Error":
                print("Error")
            if nb_colors == 1:
                A_ = nx.to_numpy_matrix(GS).T
                D = np.diagflat(1/np.sum(A_,axis=0))
                A = A_ @ D
            else:
                A = []
                for i in range(nb_colors):
                    tmp = nx.Graph((u, v, e) for u,v,e in GS.edges_iter(data=True) if e['label'] == i)
                    tmp = nx.to_numpy_matrix(tmp).T
                    D = np.diagflat(1/np.sum(tmp,axis=0))
                    tmp = tmp @ D
                    A.append(tmp)
            db_A.append((A,typ))
            for _ in range(nb_altered):
                G = self.alter_graph_struct(GS, typ, np.random.randint(max(1,int(np.floor(nb_nodes*intensity)))))
                self.alter_graph_labels(G, np.random.randint(max(1,int(np.floor(nb_nodes*intensity)))))
                if nb_colors==1:
                    A_ = nx.to_numpy_matrix(G).T
                    D = np.diagflat(1/np.sum(A_,axis=0))
                    A = A_ @ D
                else:
                    A = []
                    for i in range(nb_colors):
                        tmp = nx.Graph((u, v, e) for u,v,e in GS.edges_iter(data=True) if e['label'] == i)
                        tmp = nx.to_numpy_matrix(tmp).T
                        D = np.diagflat(1/np.sum(tmp,axis=0))
                        tmp = tmp @ D
                        A.append(tmp)
                db_A.append((A,typ))
        #np.random.shuffle(db_A)
        return np.array(db_A)

    def gen_database_test(self, nb_altered, nb_nodes, nb_colors, nb_altered_nodes_max, lap=False):
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
        db_B = [] #sans couleurs
        for typ in self.GRAPH_TYPES:
            #source graph
            GS = self.gen_graph(typ,nb_nodes,nb_colors)
            if GS == "Error":
                print("Error")
            if lap:
                A_ = nx.normalized_laplacian_matrix(GS).T
            else :
                A_ = nx.to_numpy_matrix(GS).T
            D = np.diagflat(1/np.sum(A_,axis=0))
            A = A_ @ D
            db_B.append((A,typ))
            if nb_colors >= 1:
                A = []
                for i in range(nb_colors):
                    tmp = nx.Graph(list([(u, v, e) for u,v,e in GS.edges(data=True) if e['label'] == i]))
                    for n in GS.nodes():
                        if n not in tmp.nodes():
                            tmp.add_node(n)
                    
                    if lap:
                        tmp = nx.normalized_laplacian_matrix(tmp).T
                    else :
                        tmp = nx.to_numpy_matrix(tmp).T
                    somme = np.sum(tmp,axis=0)
                    somme[np.where(somme==0)]=1 # to avoid division by zero, anyway column is 0
                    D = np.diagflat(1/somme)
                    tmp = tmp @ D
                    A.append(tmp)
                db_A.append((A,typ))
            for _ in range(nb_altered):
                G = self.alter_graph_struct(GS, typ, np.random.randint(nb_altered_nodes_max))
                self.alter_graph_labels(G, np.random.randint(nb_altered_nodes_max))
                if lap:
                    A_ = nx.normalized_laplacian_matrix(G).T
                else :
                    A_ = nx.to_numpy_matrix(G).T
                somme = np.sum(A_,axis=0)
                somme[np.where(somme==0)]=1 # to avoid division by zero, anyway column is 0
                D = np.diagflat(1/somme)
                A = A_ @ D
                db_B.append((A,typ))
                if nb_colors >= 1:
                    A = []
                    for i in range(nb_colors):
                        # for u,v,e in GS.edges(data=True):
                        #     print("e=",e)
                        tmp = nx.Graph(list([(u, v, e) for u,v,e in G.edges(data=True) if e['label'] == i]))
                        for n in G.nodes():
                            if n not in tmp.nodes():
                                tmp.add_node(n)
                        if lap:
                            tmp = nx.normalized_laplacian_matrix(tmp).T
                        else :
                            tmp = nx.to_numpy_matrix(tmp).T
                        somme = np.sum(tmp,axis=0)
                        somme[np.where(somme==0)]=1
                        D = np.diagflat(1/somme)
                        tmp = tmp @ D
                        A.append(tmp)
                    db_A.append((A,typ))
        #np.random.shuffle(db_A)
        assert len(db_A)==len(db_B)
        return np.array(db_A), np.array(db_B)

    
    
    def export_db(self, db, path):
        if not self.loaded:
            return False
        self.path = path
        with open(path,"wb") as f:
            pk.dump(db,f)
            
    def import_db(self, path):
        self.loaded = True
        self.path = path
        with open(path,"rb") as f:
            return pk.load(f)

    def gen_export_db(self, nb_graphs, nb_altered, nb_nodes, nb_colors, intensity, normalized=True):
        db = self.gen_database(nb_graphs, nb_altered, nb_nodes, nb_colors, intensity, normalized)
        path = str("dbs/db-"+str(time())+".dat")
        self.export_db(db,path)
        self.loaded = True
        self.path = path
        return (db,path)


    def load_db(self,name):
        print("Loading ",name)
        data = datasets.fetch_dataset(name, verbose=False, as_graphs=True)
        #uniques = np.unique(data.target)
        #my part
        db_A = []
        db_B = []
        print(len(data.data))
        for G,typ in zip(data.data,data.target):
            #uncolored
            G = G.get_adjacency_matrix().astype(np.float32)
            A_ = G.T
            somme = np.sum(A_,axis=0)
            somme[np.where(somme==0)]=1 # to avoid division by zero, anyway column is 0
            D = np.diagflat(1/somme)
            A = A_ @ D
            A = np.array(A,dtype=np.float32)
            typ=str(typ)
            #print(A,type(A))
            db_B.append((A,typ))
            #colored
            A = []
            # todo calculer labelisé
            # for i in range(len(np.unique(data_node_label))):
            #     tmp = nx.Graph(list([(u, v, e) for u,v,e in G.edges(data=True) if e['label'] == i]))
            #     for n in GS.nodes():
            #         if n not in tmp.nodes():
            #             tmp.add_node(n)
            #         tmp = nx.to_numpy_matrix(tmp).T
            #         somme = np.sum(tmp,axis=0)
            #         somme[np.where(somme==0)]=1 # to avoid division by zero, anyway column is 0
            #         D = np.diagflat(1/somme)
            #         tmp = tmp @ D
            #         A.append(tmp)
            #db_A.append(np.array([A,typ])
        return np.array(db_A),np.array(db_B)
