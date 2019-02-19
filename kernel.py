import numpy as np; np.random.seed(1)#probleme double graine ? 

class Kernel:

    def __init__(self, mu, N, M):
        """
        N = nombres de noeuds
        N = nombre d'aretes
        """
        self.mu = mu
        self.N = N
        self.M = M

    def raw_kernel(self, A1, A2):
        Wx = np.kron(A1,A2)
        n = Wx.shape[0]
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        m = len(Wx.nonzero()[0])
        return np.sum([self.mu(k) * qx.T @ np.power(Wx,k) @ px for k in range(self.N)])/(self.N*m)

    def build_gram_matrix(self, db):
        gram = np.empty((len(db),len(db)))
        for i in range(len(db)):
            for j in range(i+1):
                ker = self.raw_kernel(db[i],db[j])
                gram[i, j] = ker
                if i != j:
                    gram[j, i] = ker
        return gram

    #optimiser
    def build_gram_matrix_nonsq(self, X, Z):
        gram = np.empty((len(X),len(Z)))
        for i in range(len(X)):
            for j in range(len(Z)):
                ker = self.raw_kernel(X[i],Z[j])
                gram[i, j] = ker
        return gram

