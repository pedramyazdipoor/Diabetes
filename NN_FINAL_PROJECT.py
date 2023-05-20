import numpy as np
!pip install kmedoids
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import kmedoids
import random
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from numpy import save
from numpy import load
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

dim = 6
n_clusters = 40

diabetes = pd.read_csv("diabetes.csv")
diabetes = np.array(diabetes)
diabetes_normalized = preprocessing.normalize(diabetes[:,0:8],axis=0)
diabetes_normalized = diabetes_normalized.tolist()

pca = PCA()
pca.fit(diabetes_normalized)
transformed = pca.transform(diabetes_normalized)

X_train, X_test, y_train, y_test  = train_test_split(transformed[:,0:dim],diabetes[:,8] ,train_size=0.85)
save('X_test.npy', X_test)
save('y_test.npy', y_test)
save('X_train.npy', X_train)
save('y_train.npy', y_train)


#X_train = load('X_train.npy')
#X_test = load('X_test.npy')
#y_train = load('y_train.npy')
#y_test = load('y_test.npy')
X_train = X_train.tolist()


#print(diabetes_normalized[0])
#print(transformed[:,0:dim])
#print(pca.explained_variance_ratio_)

pd.DataFrame(pca.explained_variance_ratio_).plot.bar()
plt.legend('')
plt.xlabel('Principal Components')
plt.ylabel('Explained Varience')
plt.savefig("pca")

for i in range(len(X_train)):
  X_train[i].append(y_train[i])


n_data = len(X_train)
distmatrix = [[0 for i in range(0,n_data)] for j in range(0,n_data)]

def calculate_distances():
  for a in range(0,n_data):
    for b in range(0,n_data):
      distmatrix[a][b] = np.sum((X_train[a:a+1,0:dim] - X_train[b:b+1,0:dim])**2)
      
X_train = np.array(X_train)
calculate_distances()

#list = []
#for y in range(0,10):
  #start = time.time()

c = kmedoids.fasterpam(distmatrix,n_clusters,200)
  #c = kmedoids.pam(distmatrix,n_clusters,200)
  #c = kmedoids.fastpam1(distmatrix,n_clusters,200)

  #end = time.time()
  #d = end - start
  #list.append(d)

#print(np.average(list))
#print(np.std(list))


save('cluster_labels.npy', c.labels)
save('cluster_centers.npy', c.medoids)

#c.labels = load('cluster_labels.npy')
#c.medoids = load('cluster_centers.npy')

#print((c.labels))
#print("centers :",c.medoids)


phi_matrix = [[0 for i in range(0,n_clusters)] for j in range(0,n_clusters)]

cov_list = []
for u in range(0,n_clusters):
  data_tempList = []
  for v in range(0,n_data):
    if c.labels[v] == u: data_tempList.append(X_train[v:v+1,0:dim][0])
  data_tempList = np.array(data_tempList).T
  cov_list.append(np.cov(data_tempList))

cv = np.array(cov_list)
for f in range(0,n_clusters):
  det = np.linalg.det(cv[f])
  #print("det: ",det)
  if det <= 1e-5 : 
    #print(cov_list[f])
    for a in range(0,dim):
      for b in range(0,dim):
        if a==b :  cv[f][a][b] += 0.05
        else : cv[f][a][b] = 0
    #print(np.linalg.det(cov_list[f]))    

  #print(det)
#print(cv.shape)


data_centers = []
for h in range(len(c.medoids)):
  data_centers.append(X_train[c.medoids[h]])

def cal_phi(centroids):
  centroids = np.array(centroids)
  for i in range(0,n_clusters):
    for j in range(0,n_clusters):
      if i==j: phi_matrix[i][j] = 1.01
      else: 
        diff = (centroids[i:i+1,0:dim] - centroids[j:j+1,0:dim])
        power_e = np.dot(np.dot(diff,np.linalg.inv(cv[j])),diff.T)[0][0]
        #power_e = np.dot(diff,diff.T)[0][0]
        phi_matrix[i][j] = np.exp(-0.5*power_e)
  return phi_matrix      

output = []
for k in range(0,n_clusters):
  centroid_index = c.medoids[k]
  output.append(int(X_train[centroid_index][-1]))

output = np.array(output)
phi_matrix = cal_phi(data_centers)


print("phi det: ",np.linalg.det(np.array(phi_matrix)))
print(np.array(phi_matrix).shape)
print(np.array(output).shape)
w = np.dot(np.linalg.inv(phi_matrix),np.array(output))
#print(w)


def perf_measure(y_actual, y_hat):
    #TP = 0
    FP = 0
    TN = 0
    #FN = 0

    for i in range(len(y_hat)): 
        #if y_actual[i]==y_hat[i]==1:
           #TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        #if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           #FN += 1

    return(FP/(FP + TN))


def predict():
  #expected_labels = []
  output_labels = []
  centers = np.array(data_centers)
  #for t in range(0,len(test_indexes)):
    #expected_labels.append(X_train[test_indexes[t]][-1])

  for i in range(0,len(X_test)):
    sum = 0
    for j in range(0,n_clusters):
      #diff = (diabetes_normalized[test_indexes[i]:test_indexes[i]+1,0:8] - centers[j:j+1,0:8])
      diff = (X_test[i] - centers[j:j+1,0:dim])
      power_e = np.dot(np.dot(diff,np.linalg.inv(cv[j])),diff.T)[0][0]
      #power_e = np.dot(diff,diff.T)[0][0]
      #print("similarity :",np.exp(-0.5*power_e))
      sum += (np.exp(-0.5*power_e))*w[j]
    #output_labels.append(sum)  
    #output_labels.append(temp) 
    if sum >= 0.5:   output_labels.append(1)  
    else : output_labels.append(0)  
    
  
  print("accuracy :  ",accuracy_score(y_test, output_labels))
  print("FPR     :   ",perf_measure(y_test,output_labels))
  print("precision macro: ",precision_score(y_test, output_labels, average='macro',zero_division=1))
  print("recall  macro  : ",recall_score(y_test, output_labels, average='macro',zero_division=1))

predict()