from graph import gen_database, product_graph
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt 

def mu(k):
	return 1/k

#somme infinie ?
# comment choisir mu, p, q
def raw_kernel(A1, A2):
	Wx = sp.kron(A1,A2)
	n = Wx.shape[0]
	px = np.ones((n,1))/n
	qx = np.ones((n,1))/n
	return np.sum([mu(k) * qx.T @ np.power(Wx,k) @ px for k in range(1,Wx.shape[0])])

def build_gram_matrix(db):
	gram = np.empty((len(db),len(db)))
	for i in range(len(db)):
		for j in range(i+1):
			ker = raw_kernel(db[i],db[j])
			gram[i, j] = ker
			if i != j:
				gram[j, i] = ker
	return gram
	
def test(n):
	db = gen_database(n,0)
	gram = build_gram_matrix(db[:,0])
	plt.matshow(gram)
	plt.show()