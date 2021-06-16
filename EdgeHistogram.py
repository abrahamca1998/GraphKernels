import networkx as nx
import matplotlib.pyplot as plt
   

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram

# Loads the MUTAG dataset
MUTAG = fetch_dataset("NCI1", verbose=False)
G, y = MUTAG.data, MUTAG.target
G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.3, random_state=11)
X_train=np.zeros(((len(G_train)),38))
X_test=np.zeros(((len(G_test)),38))
for i in range(len(G_train)):
    graphs=list(G_train[i][2].keys())
    for j in graphs:
        X_train[i,G_train[i][2][j]]+=1
for i in range(len(G_test)):
    graphs=list(G_test[i][2].keys())
    for j in graphs:
        X_test[i,G_test[i][2][j]]+=1
K_train=np.dot(X_train,X_train.T)
K_test=np.dot(X_test,X_train.T)
SVC=SVC(kernel="precomputed",C=1)
SVC.fit(K_train,y_train)
SVC.predict(K_train)
print("Accuracy_Train:", accuracy_score(SVC.predict(K_train),y_train))
print("Accuracy_Test:", accuracy_score(SVC.predict(K_test),y_test))
print("F1-Score_Train:", f1_score(SVC.predict(K_train),y_train))
print("F1-Score_test:", f1_score(SVC.predict(K_test),y_test))
print("Precision_train:", precision_score(SVC.predict(K_train),y_train))
print("Precision_test:", precision_score(SVC.predict(K_test),y_test))
print("Recall train:", recall_score(SVC.predict(K_train),y_train))
print("Recall test:", recall_score(SVC.predict(K_test),y_test))