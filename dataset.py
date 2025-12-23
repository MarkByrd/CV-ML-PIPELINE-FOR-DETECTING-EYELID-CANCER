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

transform = transforms.Compose([
     transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(r"C:\Users\Dell\Downloads\Eyelid Cancer Dataset_2\Eyelid Cancer Dataset", transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

print(dataset.__len__())

#img, label = dataset[15]
#plt.imshow(img.permute(1, 2, 0))
#plt.axis("off")
#plt.show()

#cnn_feat = torchvision.models.resnet50()
#cnn_feat = torch.nn.Sequential(*list(cnn_feat.children())[:-1])
#cnn_feat.eval()

out = []
i = 1

#with torch.no_grad():
#    for img, label in iter(loader):
#        out.append(list(np.append(cnn_feat(img).numpy().flatten(), label)))
#        print(len(out))
#        print(i)
#        i+=1
    

#data = np.array(out)
#np.savetxt("features.csv", data, delimiter=",")

def chan_means(tens):
    results = []
    for i in range(3):
        results.append(np.mean(tens.numpy()[:,i,:,:]))
    return results
out = []
i = 1

for img, label in iter(loader):
    x = list(chan_means(img))
    x.append(label.item())
    out.append(x)
    i+=1
print(out)
data = np.array(out)
np.savetxt("channel_averages.csv", data, delimiter=",")
