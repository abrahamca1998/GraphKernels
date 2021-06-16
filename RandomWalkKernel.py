
import networkx as nx
import matplotlib.pyplot as plt
   

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram
def function_graph(G,item):
    Graph= nx.Graph()
    nodes=sorted(G[item][1].keys())
    edges=sorted(G[item][0])
    edges_labels=G[item][2]
    edges_colors=np.zeros([len(edges_labels),])
    for i,j in enumerate((edges_labels.keys())):
        edges_colors[i]=(float(edges_labels[j]))
    nodes_colors=(np.array(list(G[item][1].values())))
    Graph.add_edges_from(edges)
    Graph.add_nodes_from(nodes)
    nx.draw(Graph,edgelist=edges,edge_color=edges_colors,node_color=nodes_colors,with_labels=True, cmap=plt.cm.Reds,edge_cmap=plt.cm.Blues)
    return edges,nodes,nodes_colors
def graph_product(nodes_1,nodes_2,edges_1,edges_2,nodes_labels1,nodes_labels2):
    nodes_joined=[]
    for i in range(len(nodes_1)):
        for j in range(len(nodes_2)):
            if nodes_labels1[i]==nodes_labels2[j]:
                nodes_joined.append((nodes_1[i],nodes_2[j]))
    edges_joined=[]
    for node in nodes_joined:
        element1=node[0]
        element2=node[1]
        edges_1=np.array(edges_1)
        edges_2=np.array(edges_2)
        a=edges_1[edges_1[:,0]==element1]
        b=edges_2[edges_2[:,0]==element2]
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                if ((a[i][0],b[j][0]) in nodes_joined and (a[i][1],b[j][1]) in nodes_joined):
                    edges_joined.append(((a[i][0],b[j][0]),(a[i][1],b[j][1]))) 
    return nodes_joined,edges_joined
def graph_product_nw(nodes_1,nodes_2,edges_1,edges_2,nodes_labels1,nodes_labels2):
    Graph1= nx.Graph()
    Graph1.add_edges_from(edges_1)
    Graph1.add_nodes_from(nodes_1)
    Graph2= nx.Graph()
    Graph2.add_edges_from(edges_2)
    Graph2.add_nodes_from(nodes_2)
    product=nx.tensor_product(Graph1, Graph2)
    nodes_joined=list(product.nodes)
    edges_joined=list(product.edges)
    return nodes_joined,edges_joined
    
    
def compute_kernel(nodes_joined,edges_joined):
    dict_nodes = dict(zip(nodes_joined,range(len(nodes_joined))))
    adjacency_matrix=np.zeros([len(nodes_joined),len(nodes_joined)])
    for edge in edges_joined:
        adjacency_matrix[dict_nodes[edge[0]],dict_nodes[edge[1]]]=1
    lambda_max=np.max(np.linalg.eigvals(adjacency_matrix)).real
    lambda_max=lambda_max-(1/10)*lambda_max
    lambda_reg=1/lambda_max
    e_vec=np.ones([1,len(adjacency_matrix)])
    I=np.identity(len(adjacency_matrix))
    K=np.dot(np.dot(e_vec,np.linalg.inv(I-lambda_reg*adjacency_matrix)),e_vec.T)
    return K[0][0]
def compute_kernel_matrix(G):
    len_data=len(G)
    K_max=np.zeros([len_data,len_data])
    for i in range(len_data):
        for j in range(len_data):
            print("i:",i,"j:",j)
            edges_1,nodes_1,nodes_labels1=function_graph(G,i)
            edges_2,nodes_2,nodes_labels2=function_graph(G,j)
            nodes_joined,edges_joined=graph_product_nw(nodes_1,nodes_2,edges_1,edges_2,nodes_labels1,nodes_labels2)
            K=compute_kernel(nodes_joined,edges_joined)
            K_max[i,j]=K

MUTAG = fetch_dataset("NCI1", verbose=False)
G, y = MUTAG.data, MUTAG.target
G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.3, random_state=7)

K_train=compute_kernel_matrix(G_train)
K_test=compute_kernel_matrix(G_test)

SVC=SVC(kernel="precomputed",C=100,decision_function_shape='ovo')
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