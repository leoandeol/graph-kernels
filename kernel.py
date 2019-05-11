import numpy as np; np.random.seed(1)#probleme double graine ?
from scipy.sparse.linalg import cg
from scipy.linalg import solve_discrete_lyapunov, eigh
from scipy.optimize import fixed_point
from control import dlyap
from time import time

#coder graph labélisés
class Kernel:

    def __init__(self, lbd, N, k=None):
        self.lbd = lbd
        self.N = N
        self.mu = lambda x: pow(self.lbd,x)
        # k is used in different kernels differently, usually number of loops of algorithms
        self.k = k
        # computation time
        self.comp_time = 0

    def kron(self, A1, A2):
        if type(A1)==np.matrix:
            return np.kron(A1,A2)
        else:
            Wx = np.kron(A1[0],A2[0])
            for i in range(1,len(A1)):
                Wx = Wx + np.kron(A1[i],A2[i])
            return Wx

    def raw_kernel(self, A1, A2):
        Wx = self.kron(A1,A2)
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        return np.sum([self.mu(k) * qx.T @ np.power(Wx,k) @ px for k in range(self.N)])/n

    def inv_kernel(self, A1, A2):
        # (16)
        Wx = self.kron(A1,A2)
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        return qx.T @ np.linalg.inv(np.identity(n)-self.lbd*Wx)@px

    #ne fonctionne pas quand matrices singulieres
    def sylv_eq_kernel(self, A1, A2):
        """ 
        Sylvester equation Methods (lyapunov)
        O(n^3)
        for graph with no labels only : else implement "COMPUTATION OF THE CANONICAL DECOMPOSITION BYMEANS OF A SIMULTANEOUS GENERALIZED SCHURDECOMPOSITION∗"
        with https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.linalg.qz.html
        """
        #https://python-control.readthedocs.io/en/0.8.0/generated/control.dlyap.html
        #resoudre AXQt - X + C
        n = A1.shape[0]*A2.shape[0]
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        M = dlyap(A=A1,Q=A2,C=px.reshape((A1.shape[0],A2.shape[0])))
        return qx.T @ M.reshape((-1,1))
        

    def conj_grad_kernel(self, A1, A2):
         """ 
         Conjugate Gradient Methods
         O(n^3)
         for graph with discrete edge labels
         """
         Wx = self.kron(A1,A2)
         n = Wx.shape[0]
         I = np.identity(n)
         M = I-self.lbd*Wx
         assert M.shape==Wx.shape
         px = np.ones((n,1))/self.N
         qx = np.ones((n,1))/self.N
         # donner M essentiel simplifier M=inv(M)
         v = np.random.randint(0,n,size=(n,1))
         x,_ = cg(M,px,x0=px,M=v@v.T)
         return qx.T@x
     
    def fixed_point_kernel(self, A1, A2):
        Wx = self.kron(A1,A2)
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        #diagonaliser
        Wx = (Wx + Wx.T)/2


        
        #on calcule que la valeur propre max
        #self.lbd = 1/(12+abs(eigh(Wx,eigvals_only=True,eigvals=(n-1,n-1))[0]))
        #si lambda trop proche de l'inverse de la valeur propre peut etre tres lent à converger quand la matrice d'adjacence est très dense

        
        func = lambda x: np.asarray(px+(self.lbd*Wx)@x)
        x = fixed_point(func, np.asarray(px),maxiter=1500)
        k = np.real(np.asscalar(qx.T@x))
        return k

    def spec_decomp_kernel(self, A1, A2):
        Wx = self.kron(A1,A2)
        #diagonaliser Wx
        #Wx = (Wx + Wx.T)/2
        Dx,Px = np.linalg.eig(Wx)
        # real = np.isreal(Dx)
        # print("Px shape before",Px.shape)
        # Dx = Dx[np.where(real==True)]
        # Px = np.delete(Px, np.where(real==False),axis=0)
        # Px = np.delete(Px, np.where(real==False),axis=1)
        # print("Px shape after",Px.shape)


        # matrice singuliere pour graphe staR ?
        if np.linalg.det(Px)==0:
            return 
        
        Dx = np.diag(Dx)
        Px1 = np.linalg.inv(Px)

        
        n = Dx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/self.N
        qx = np.ones((n,1))/self.N
        k = qx.T @ Px @ np.linalg.inv(np.identity(n)-self.lbd*Dx) @ Px1 @ px
        k = np.real(np.asscalar(k))
        return k
    
    def build_gram_matrix(self, db, kernel):
        self.comp_time = time()
        gram = np.empty((len(db),len(db)))
        for i in range(len(db)):
            for j in range(i+1):
                ker = kernel(db[i],db[j])
                gram[i, j] = ker
                if i != j:
                    gram[j, i] = ker
        self.comp_time = time() - self.comp_time
        return gram

    #optimiser
    def build_gram_matrix_nonsq(self, X, Z, kernel):
        gram = np.empty((len(X),len(Z)))
        for i in range(len(X)):
            for j in range(len(Z)):
                ker = kernel(X[i],Z[j])
                gram[i, j] = ker
        return gram

