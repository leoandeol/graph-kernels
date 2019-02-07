from graph import gen_database, product_graph
import matplotlib.pyplot as plt


import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import zero_one_loss

def mu(k):
	return pow(0.9,k)

#somme infinie ?
# comment choisir mu, p, q
def raw_kernel(A1, A2):
	Wx = np.kron(A1,A2)
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

#optimiser
def build_gram_matrix_nonsq(X, Z):
	gram = np.empty((len(X),len(Z)))
	for i in range(len(X)):
		for j in range(len(Z)):
			ker = raw_kernel(X[i],Z[j])
			gram[i, j] = ker
	return gram
	
def test(db):
        n = int(len(db)*0.7)
        X, y = shuffle(db[:,0], db[:,1])
        X_train, X_test = X[:n], X[n:]
        y_train, y_test = y[:n], y[n:]
        svc = SVC(kernel='precomputed')
        kernel_train = build_gram_matrix(X_train)
        sns.heatmap(kernel_train)
        svc.fit(kernel_train, y_train)
        kernel_test = build_gram_matrix_nonsq(X_test, X_train.T)
        y_pred = svc.predict(kernel_test)
        print(zero_one_loss(y_test,y_pred))
