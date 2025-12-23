import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
import sklearn
import tqdm
import time

data = np.genfromtxt('features.csv', delimiter=',')
data_2 = np.genfromtxt("channel_averages.csv", delimiter = ",")
model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, random_state=42)

def cross_val(k, data):
    np.random.shuffle(data)
    errors = []
    L = len(data)//k
    for j in range(k):
        model = sklearn.ensemble.RandomForestClassifier(random_state= 69)
        test_data = data[L*j:L*(j+1),:]
        train_data = np.vstack([data[0:L * j,:], data[L * (j + 1):,:]])
        model.fit(train_data[:,:-1], train_data[:,-1])
        y_pred = model.predict(test_data[:,:-1])
        err = sklearn.metrics.accuracy_score(y_pred, test_data[:,-1])
        print(err)
        errors.append(err)
    return np.mean(errors)
def image_cross_val(k, dataset):
    

results = []
print(cross_val(10, data_2))
print(len(data_2))
k = 0
for i in range(len(data)):
    for j in range(i+1,len(data)):
        if (data[i] == data[j]).all():
            print("Duplicate found")
            k +=1
print(f"{k} total duplicates")

#N = 20
#for i in range(N):
#    results.append(cross_val(10, data))
#print(np.mean(results))
#fig, ax = plt.subplots()
#ax.hist(results, bins = 8)
#plt.show()
