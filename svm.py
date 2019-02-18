import seaborn as sns; sns.set()

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import zero_one_loss
from kernel import Kernel

class SVM:

    def __init__(self, data, ratio_split, mu):
        self.n = int(len(db)*0.7)
        self.N = np.max(db[:,0].shape[0])
        self.M = np.max([len(x.nonzero()[0]) for x in db[:,0]])
        self.k = Kernel(N,M)
        X, y = db[:,0], db[:,1]
        #shuffle turned of for debug here and in graph
        #X, y = shuffle(db[:,0], db[:,1])
        self.X_train, self.X_test = X[:n], X[n:]
        self.y_train, self.y_test = y[:n], y[n:]
        self.svc = SVC(kernel='precomputed')

    def learn(self):
        self.kernel_train = self.k.build_gram_matrix(self.X_train)
        self.svc.fit(self.kernel_train, self.y_train)

    def score(self):
        #check is matrice dans bon sens
        self.kernel_test = self.k.build_gram_matrix_nonsq(self.X_test, self.X_train.T)
        self.y_pred = self.svc.predict(self.kernel_test)
        return zero_one_loss(y_test,y_pred)

    def display_heatmap(categ = "train"):
        #revoir labels
        if categ.lower() == "train":
            sns.heatmap(kernel_train,xticklabels=y_train,yticklabels=y_train)
        elif categ.lower() == "test":       
            sns.heatmap(kernel_test,xticklabels=y_train,yticklabels=y_train)
      
