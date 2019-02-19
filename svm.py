import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import zero_one_loss
from kernel import Kernel
import numpy as np

class SVM:

    def __init__(self, db, ratio_split, mu):
        self.n = int(len(db)*0.7)
        self.N = np.max(db[:,0].shape[0])
        self.M = np.max([len(x.nonzero()[0]) for x in db[:,0]])
        self.mu = mu
        self.k = Kernel(self.mu,self.N,self.M)
        X, y = db[:,0], db[:,1]
        #shuffle turned of for debug here and in graph
        #X, y = shuffle(db[:,0], db[:,1])
        self.X_train, self.X_test = X[:self.n], X[self.n:]
        self.y_train, self.y_test = y[:self.n], y[self.n:]
        self.svc = SVC(kernel='precomputed')

    def learn(self):
        self.kernel_train = self.k.build_gram_matrix(self.X_train)
        self.svc.fit(self.kernel_train, self.y_train)

    def score(self):
        #check is matrice dans bon sens
        self.kernel_test = self.k.build_gram_matrix_nonsq(self.X_test, self.X_train.T)
        self.y_pred = self.svc.predict(self.kernel_test)
        return zero_one_loss(self.y_test,self.y_pred)

    def display_heatmap(self, categ = "traintest"):
        #revoir labels
        if "train" in categ.lower():
            plt.figure()
            sns.heatmap(self.kernel_train,xticklabels=self.y_train,yticklabels=self.y_train).set_title("Train")
        if "test" in categ.lower():
            plt.figure()
            sns.heatmap(self.kernel_test,xticklabels=self.y_train,yticklabels=self.y_train).set_title("Test")
      
