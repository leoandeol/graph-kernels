from graph import gen_database, product_graph
import scipy.sparse as sp
import matplotlib.pyplot as plt


import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import zero_one_loss

def mu(k):
	return 1

#somme infinie ?
# comment choisir mu, p, q
def raw_kernel(A1, A2):
	Wx = sp.kron(A1,A2)
	n = Wx.shape[0]
	px = np.ones((n,1))/n
	qx = np.ones((n,1))/n
	return np.sum([mu(k) * qx.T @ np.power(Wx,k) @ px for k in range(1,5*Wx.shape[0])])

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
        #db = gen_database()
        gram = build_gram_matrix(db[:,0])
        #plt.matshow(gram)
        sns.heatmap(gram)
        plt.show()
        n = int(gram.shape[0]*0.7)
        X, y = shuffle(db[:,0], db[:,1])
        print(X.shape, y.shape)
        X_train, X_test = X[:n], X[n:]
        y_train, y_test = y[:n], y[n:]
        svc = SVC(kernel='precomputed')
        
        kernel_train = build_gram_matrix(X_train)
        svc.fit(kernel_train, y_train)
        # en fait np.dot(X_test, X_train.T)
        kernel_test = build_gram_matrix_nonsq(X_test, X_train.T)
        y_pred = svc.predict(kernel_test)
        print(zero_one_loss(y_test,y_pred))
        return svc
