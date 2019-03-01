import numpy as np; np.random.seed(1)#probleme double graine ?
from scipy.sparse.linalg import cg

class Kernel:

    def __init__(self, lbd, N, M):
        """
        N = nombres de noeuds
        N = nombre d'aretes
        """
        self.lbd = lbd
        self.N = N
        self.M = M
        self.mu = lambda x: pow(self.lbd,x)

    def raw_kernel(self, A1, A2):
        Wx = np.kron(A1,A2)
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        return np.sum([self.mu(k) * qx.T @ np.power(Wx,k) @ px for k in range(self.N)])/m

    def inv_kernel(self, A1, A2):
        # (16)
        Wx = np.kron(A1,A2)
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        return qx.T @ np.linalg.inv(np.identity(n)-self.lbd*Wx)@px /m

    def sylv_eq_kernel(self, A1, A2):
        """ 
        Sylvester equation Methods
        O(n^3)
        for graph with discrete edge labels
        """
        pass

    #TODO optimiser tehcnique y
    def conj_grad_kernel(self, A1, A2):
         """ 
         Conjugate Gradient Methods
         O(n^3)
         for graph with discrete edge labels
         """
         Wx = np.kron(A1,A2)
         n = Wx.shape[0]
         I = np.identity(n)
         M = I-self.lbd*Wx
         px = np.ones((n,1))/self.N
         qx = np.ones((n,1))/self.N
         x,_ = cg(M,px)
         return qx.T@x

    def build_gram_matrix(self, db, kernel):
        gram = np.empty((len(db),len(db)))
        for i in range(len(db)):
            for j in range(i+1):
                ker = kernel(db[i],db[j])
                gram[i, j] = ker
                if i != j:
                    gram[j, i] = ker
        return gram

    #optimiser
    def build_gram_matrix_nonsq(self, X, Z, kernel):
        gram = np.empty((len(X),len(Z)))
        for i in range(len(X)):
            for j in range(len(Z)):
                ker = kernel(X[i],Z[j])
                gram[i, j] = ker
        return gram

