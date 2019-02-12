from graph import gen_database, product_graph
import matplotlib.pyplot as plt


import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import zero_one_loss

def mu(k):
    return pow(0.8,k)

#somme infinie ?
# comment choisir mu, p, q
# n constant
#max n de la base
N = 0
M = 0

def raw_kernel(A1, A2):
    Wx = np.kron(A1,A2)
    n = Wx.shape[0]
    px = np.ones((n,1))/N
    qx = np.ones((n,1))/N
    m = len(Wx.nonzero()[0])
    return np.sum([mu(k) * qx.T @ np.power(Wx,k) @ px for k in range(N)])/(N*m)

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
    global N
    global M
    n = int(len(db)*0.7)
    N = np.max(db[:,0].shape[0])
    M = np.max([len(x.nonzero()[0]) for x in db[:,0]])
    print("max N",N)
    print("max M",M)
    X, y = db[:,0], db[:,1]
    #X, y = shuffle(db[:,0], db[:,1])
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    svc = SVC(kernel='precomputed')
    kernel_train = build_gram_matrix(X_train)
    sns.heatmap(kernel_train,xticklabels=y_train,yticklabels=y_train)
    svc.fit(kernel_train, y_train)
    kernel_test = build_gram_matrix_nonsq(X_test, X_train.T)
    #sns.heatmap(kernel_test)
    y_pred = svc.predict(kernel_test)
    print(zero_one_loss(y_test,y_pred))
