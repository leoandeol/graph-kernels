import numpy as np; np.random.seed(1)#probleme double graine ?
from scipy.sparse.linalg import cg
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import fixed_point

#coder graph labélisés
class Kernel:

    def __init__(self, lbd, N, M):
        self.lbd = lbd
        self.N = N
        self.mu = lambda x: pow(self.lbd,x)

    #pourquoi diviser par self.N
    def raw_kernel(self, A1, A2):
        Wx = np.kron(A1,A2)
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        #diviser par le nombre d'arretes le rapproche du reste et augmente le score?
        return np.sum([self.mu(k) * qx.T @ np.power(Wx,k) @ px for k in range(self.N)])/m

    def inv_kernel(self, A1, A2):
        # (16)
        Wx = np.kron(A1,A2)
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        return qx.T @ np.linalg.inv(np.identity(n)-self.lbd*Wx)@px

    def sylv_eq_kernel(self, A1, A2):
        """ 
        Sylvester equation Methods (lyapunov)
        O(n^3)
        for graph with no labels only 
        """
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        #introuvable
        

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
         assert M.shape==Wx.shape
         px = np.ones((n,1))/self.N
         qx = np.ones((n,1))/self.N
         # donner M essentiel
         x,_ = cg(M,px,x0=px,M=np.linalg.inv(M))
         return qx.T@x
     
    def fixed_point_kernel(self, A1, A2):
        Wx = np.kron(A1,A2)
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        #ca marche mais cuisine
        self.lbd = 1/(1+np.max(np.linalg.eigvals(Wx)))
        func = lambda x: np.asarray(px+self.lbd*Wx@x)
		x = fixed_point(func, np.asarray(px),maxiter=1000)
		try:
            x = fixed_point(func, np.asarray(px),maxiter=1000)
        except:
            print("somme en ligne",np.sum(Wx,axis=1))
            print("somme en colonne",np.sum(Wx,axis=0))
            print(Wx.shape,"det=",np.linalg.det(self.lbd*Wx))
        k = np.real(np.asscalar(qx.T@x))
        return k

    def spec_decomp_kernel(self, A1, A2):
        Wx = np.kron(A1,A2)
        Dx,Px = np.linalg.eig(Wx)
        real = np.isreal(Dx)
        print("Px shape before",Px.shape)
        Dx = Dx[np.where(real==True)]
        Px = np.delete(Px, np.where(real==False),axis=0)
        Px = np.delete(Px, np.where(real==False),axis=1)
        print("Px shape after",Px.shape)

        Dx = np.diag(Dx)
        Px1 = np.linalg.inv(Px)
        print("Dx shape",Dx.shape)
        n = Dx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        k = qx.T @ Px @ np.linalg.inv(np.identity(n)-self.lbd*Dx) @ Px1 @ px
        k = np.real(np.asscalar(k))
        print(k)
        print(type(k))
        return k
    
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

