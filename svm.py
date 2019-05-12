import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from kernel import Kernel
import numpy as np
from time import time

class SVM:

    def __init__(self, db, ratio_split, lbd, kernel="raw_kernel"):
        self.n = int(len(db)*ratio_split)
        self.lbd = lbd
        self.k = Kernel(self.lbd)
        self.ker = getattr(self.k, kernel)
        self.X, self.y = db[:,0], db[:,1]#shuffle(db[:,0], db[:,1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(db[:,0], db[:,1], train_size=ratio_split, random_state=42, stratify=db[:,1])
        self.svc = SVC(kernel='precomputed')

    def learn(self):
        self.kernel_train = self.k.build_gram_matrix(self.X_train, self.ker)
        self.svc.fit(self.kernel_train, self.y_train)
        return self.k.comp_time

    def score(self):
        #check is matrice dans bon sens
        self.kernel_test = self.k.build_gram_matrix_nonsq(self.X_test, self.X_train.T, self.ker)
        self.y_pred = self.svc.predict(self.kernel_test)
        return 1-zero_one_loss(self.y_test,self.y_pred)

    def cross_val_score(self, k):
        start = time()
        self.kernel = self.k.build_gram_matrix(self.X, self.ker)
        end = time() - start
        score = cross_val_score(self.svc, self.kernel, self.y, cv=k)
        return { "accuracy": sum(score)/len(score), "time": end, "stddev": np.std(score)}

    def display_heatmap(self, categ = "traintest"):
        #revoir labels
        if "train" in categ.lower():
            plt.figure()
            sns.heatmap(self.kernel_train,xticklabels=self.y_train,yticklabels=self.y_train).set_title("Train")
        if "test" in categ.lower():
            plt.figure()
            sns.heatmap(self.kernel_test,xticklabels=self.y_train,yticklabels=self.y_train).set_title("Test")
      
