import numpy as np; np.random.seed(1)
from scipy.linalg import solve_discrete_lyapunov, eigh
from scipy.optimize import fixed_point
from control import dlyap
from math import exp
from time import time
from scipy.sparse.linalg import svds


#coder graph labélisés
class Kernel:

    def __init__(self, lbd, k=None):
        self.lbd = lbd
        self.mu = lambda x: pow(self.lbd,x)
        # k is used in different kernels differently, usually number of loops of algorithms
        self.k = k
        # computation time
        self.comp_time = 0

    def kron(self, A1, A2):
        if type(A1)==np.matrix or type(A1)==np.ndarray:
            return np.kron(A1,A2)
        else:
            Wx = np.kron(A1[0],A2[0])
            for i in range(1,len(A1)):
                Wx = Wx + np.kron(A1[i],A2[i])
            return Wx

    def shape(self, A):
        if type(A)==np.matrix or type(A)==np.ndarray:
            return A.shape
        else:
            return A[0].shape

    def conjugate_grad(self,A, b, x=None):
        """
        Description
        -----------
        Solve a linear equation Ax = b with conjugate gradient method.
        Parameters
        ----------
        A: 2d numpy.array of positive semi-definite (symmetric) matrix
        b: 1d numpy.array
        x: 1d numpy.array of initial point
        Returns
        -------
        1d numpy.array x such that Ax = b
        Source : https://gist.github.com/sfujiwara/b135e0981d703986b6c2
        """
        n = len(b)
        if x is None:
            x = np.ones(n)
        r = np.dot(A, x) - b
        p = - r
        r_k_norm = np.dot(r.T, r)
        for i in range(self.k):
            Ap = np.dot(A, p)
            alpha = r_k_norm / np.dot(p.T, Ap)
            alpha = np.asscalar(alpha)
            x += alpha * p
            r += alpha * Ap
            r_kplus1_norm = np.dot(r.T, r)
            beta = r_kplus1_norm / r_k_norm
            beta = np.asscalar(beta)
            r_k_norm = r_kplus1_norm
            if r_kplus1_norm < 1e-5:
                #print('Itr:', i+1)
                break
            p = beta * p - r
        return x

    def raw_kernel(self, A1, A2):
        Wx = self.kron(A1,A2)
        n = Wx.shape[0]
        px = np.ones((n,1))/n
        qx = np.ones((n,1))/n
        return np.sum([self.mu(k) * qx.T @ np.power(Wx,k) @ px for k in range(self.N)])/(self.N)

    def inv_kernel(self, A1, A2):
        # (16)
        Wx = self.kron(A1,A2)
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/n
        qx = np.ones((n,1))/n
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
        n1 = A1.shape[0]
        n2 = A2.shape[0]
        #in case the matrix is empty i add a diagonal, works great
        A1 = (A1+A1.T)/2 + np.eye(n1)*1e1
        A2 = (A2+A2.T)/2 + np.eye(n2)*1e1
        n = A1.shape[0]*A2.shape[0]
        px = np.ones((n,1))/n
        qx = np.ones((n,1))/n
        M = dlyap(A=self.lbd*A1,Q=A2,C=px.reshape((A1.shape[0],A2.shape[0])))
        return -1*np.asscalar(qx.T @ M.reshape((-1,1)))
        

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
         px = np.ones((n,1))/n
         qx = np.ones((n,1))/n
         # donner M essentiel simplifier M=inv(M)
         v = np.random.randint(0,n,size=(n,1))
         #x,_ = cg(M,px,x0=px,M=v@v.T,maxiter=self.k)
         x = self.conjugate_grad(M,px,px)
         return qx.T@x
     
    def fixed_point_kernel(self, A1, A2):
        Wx = self.kron(A1,A2)
        n = Wx.shape[0]
        m = len(Wx.nonzero()[0])
        px = np.ones((n,1))/n
        qx = np.ones((n,1))/n
        #diagonaliser
        Wx = (Wx + Wx.T)/2

        # if self.lbd >= 1/abs(eigh(Wx,eigvals_only=True,eigvals=(n-1,n-1))[0]):
        #     print("Cannot converge")
        #     raise ValueError()

        
        func = lambda x: np.asarray(px+(self.lbd*Wx)@x)
        #x = fixed_point(func, np.asarray(px),maxiter=1500)

        x0 = np.asarray(px)
        itera = self.k if self.k is not None else 1000
        last = -99999999999
        x = x0
        for i in range(itera):
            last = x
            x = func(x)
            if np.linalg.norm(x-last)<=1e-5:
                break
        
        k = np.real(np.asscalar(qx.T@x))
        return k

    def spec_decomp_kernel(self, A1, A2):
        # Wx = self.kron(A1,A2)
        # #diagonaliser Wx
        # #Wx = (Wx + Wx.T)/2
        # Dx,Px = np.linalg.eig(Wx)
        # # real = np.isreal(Dx)
        # # print("Px shape before",Px.shape)
        # # Dx = Dx[np.where(real==True)]
        # # Px = np.delete(Px, np.where(real==False),axis=0)
        # # Px = np.delete(Px, np.where(real==False),axis=1)
        # # print("Px shape after",Px.shape)


        # # matrice singuliere pour graphe staR ?
        # if np.linalg.det(Px)==0:
        #     return 
        
        # Dx = np.diag(Dx)
        # Px1 = np.linalg.inv(Px)

        
        # n = Dx.shape[0]
        # m = len(Wx.nonzero()[0])
        # px = np.ones((n,1))/self.N
        # qx = np.ones((n,1))/self.N
        # k = qx.T @ Px @ np.linalg.inv(np.identity(n)-self.lbd*Dx) @ Px1 @ px
        # k = np.real(np.asscalar(k))
        n1 = A1.shape[0]
        n2 = A2.shape[0]
        #in case the matrix is empty i add a diagonal, works great
        A1 = (A1+A1.T)/2 + np.eye(n1)
        A2 = (A2+A2.T)/2 + np.eye(n2)
        # check_sym = lambda a : np.allclose(a, a.T, rtol=1e-05, atol=1e-08)
        # assert check_sym(A1)==True
        # assert check_sym(A2)==True
        n1 = A1.shape[0]
        n2 = A2.shape[0]
        D1,P1 = np.linalg.eig(A1)
        D2,P2 = np.linalg.eig(A2)

        if not np.all(np.isreal(D1)) or not np.all(np.isreal(D2)):
            return 0
        
        D1 = np.diag(D1)
        D2 = np.diag(D2)
        Pinv1 = np.linalg.inv(P1)
        Pinv2 = np.linalg.inv(P2)
        
        p1 = np.ones((n1,1))/n1
        q1 = np.ones((n1,1))/n1
        p2 = np.ones((n2,1))/n2
        q2 = np.ones((n2,1))/n2
        
        # always using an exponential kernel
        part1 = self.kron(q1.T@P1,q2.T@P2)
        part2 = 1/np.diag(np.identity(n1*n2)-self.lbd*np.kron(D1,D2))
        part2[np.isnan(part2)]=0
        part2=np.diag(part2)
        part3 = self.kron(Pinv1@p1,Pinv2@p2)
        k = part1 @ part2 @part3
        return k
    
    def build_gram_matrix(self, db, kernel, nkp=False):
        try:
            self.N = np.max([x.shape[0] for x in db])
        except:
            self.N = np.max([x[0].shape[0] for x in db])
        self.comp_time = time()
        gram = np.empty((len(db),len(db)))
        for i in range(len(db)):
            for j in range(i+1):
                A = db[i] # np.copy() ?
                B = db[j]
                if nkp:
                    A,B,_ = self.nkp(self.kron(A,B), self.shape(A), self.shape(B), hermit=False)
                ker = kernel(A,B)
                gram[i, j] = ker
                if i != j:
                    gram[j, i] = ker
        self.comp_time = time() - self.comp_time
        # Normalisation
        gram -= np.ones(gram.shape)*np.min(gram)
        gram /= np.max(gram)
        return gram

    #optimiser
    def build_gram_matrix_nonsq(self, X, Z, kernel):
        self.N = np.max([x.shape[0] for x in X])
        gram = np.empty((len(X),len(Z)))
        for i in range(len(X)):
            for j in range(len(Z)):
                A = X[i]
                B = Z[j]
                ker = kernel(A,B)
                gram[i, j] = ker
        gram -= np.ones(gram.shape)*np.min(gram)
        gram /= np.max(gram)
        return gram


    def scale_and_compare(self,M1,M2,normalize=True):
        if normalize == True:
            M1 = np.copy(M1)
            M1 -= np.ones(M1.shape)*np.min(M1)
            M1 /= np.max(M1)
            M2 = np.copy(M2)
            M2 -= np.ones(M2.shape)*np.min(M2)
            M2 /= np.max(M2)
        return np.linalg.norm(M1-M2)/(M1.shape[0]*M1.shape[1])

    def nkp(self, A, bdim, cdim, hermit=False):
        A = np.asarray(A)
        m = A.shape[0]
        n = A.shape[1]
        m1 = bdim[0]
        n1 = bdim[1]
        m2 = cdim[0]
        n2 = cdim[1]
        
        assert m == m1 * m2
        assert n == n1 * n2
        
        if hermit:
            assert m1 == n1
            assert m2 == n2
            A = 0.5 * ( A + A.T)
        
        R = A.reshape((m2, m1, n2, n1))
        R = np.transpose(R, (1, 3, 0, 2))
        R = R.reshape((m1*n1, m2*n2))
        
        B, S, C  =  svds(R,1)
        
        SqrtS = np.sqrt(np.asscalar(S))
        
        B = (B * SqrtS).reshape((m1, n1))
        C = (C * SqrtS).reshape((m2, n2))
        
        if hermit:
            B = 0.5 * ( B + B.T)
            C = 0.5 * ( C + C.T)
            
            if np.all(np.diag( B ) < 0) and np.all(np.diag( C ) < 0):
                B = -B
                C = -C
                
        D = A - np.kron(B,C)
        if hermit:
            D = 0.5 * ( D + D.T)
        return B, C, np.linalg.norm(D)/(m*n)
