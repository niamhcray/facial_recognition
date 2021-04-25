#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:13:53 2021

@author: niamhcray
"""

import numpy as np
import matplotlib.pyplot as plt
import os, time

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms as transforms

# constants
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
BATCH_SIZE = 48

def plot_portraits(images, titles, h, w, n_row, n_col):
    ''' enables you to visualise the images'''
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

#load data 
    
dir = 'lfwcrop_grey/faces'
celebrity_photos=os.listdir(dir) 

for image in celebrity_photos:
    if image == '.DS_Store':
        celebrity_photos.remove('.DS_Store')

celebrity_images=[dir+'/'+ photo for photo in celebrity_photos]
images = np.array([plt.imread(image) for image in celebrity_images], dtype=np.float64)
celebrity_names=[name[:name.find('0')-1].replace("_", " ") for name in celebrity_photos]
n_samples, h, w = images.shape
X = images.reshape(n_samples, h*w)

le = LabelEncoder()
le.fit(celebrity_names)
y = le.transform(celebrity_names)
n_classes = len(le.classes_)
target_names = le.classes_
n_features = X.shape[1]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_classes: %d" % n_classes)

# Normalize images to lie in [-1,1]
for i in range(n_samples):
    images[i,:] = images[i,:]/abs(images[i,:]).max() #normalising pixels to be in the range [-1,1]
    
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(y.reshape(-1,1))  
'''
dir = 'testimages/zoomPIL'
testcelebrity_photos=os.listdir(dir) 

for image in testcelebrity_photos:
    if image == '.DS_Store':
        testcelebrity_photos.remove('.DS_Store')

testcelebrity_images=[dir+'/'+ photo for photo in testcelebrity_photos]
testcelebrity_images = sorted(testcelebrity_images, key=lambda x: float(x[19:-4]))
testimages = np.array([plt.imread(image) for image in testcelebrity_images], dtype=np.float64)
testcelebrity_names=[name[:name.find('0')-1].replace("_", " ") for name in testcelebrity_photos]
testn_samples, h, w = testimages.shape
Xtest = testimages.reshape(testn_samples, h*w) 
'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15 , random_state=42, stratify=y)
print('number of training images:', len(y_train))
print('number of test images:', len(y_test))
print('number of features: ', X.shape[1])

train_names = target_names[y_train]
test_names = target_names[y_test]

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
train_data = data_utils.TensorDataset(X_train, y_train)
test_data = data_utils.TensorDataset(X_test, y_test)

#Xtest = torch.from_numpy(Xtest.astype(np.float32))
#test_data = data_utils.TensorDataset(Xtest, y_test)


trainloader = data_utils.DataLoader(
    train_data,
    batch_size = BATCH_SIZE,
    shuffle = False #shuffles the data at every epoch
)

testloader = data_utils.DataLoader(
    test_data,
    batch_size = BATCH_SIZE,
)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=n_features, out_features=300)
        self.enc2 = nn.Linear(in_features=300, out_features=250)
        self.enc3 = nn.Linear(in_features=250, out_features=200)
        # decoder 
        self.dec2 = nn.Linear(in_features=200, out_features=250)
        self.dec3 = nn.Linear(in_features=250, out_features=300)
        self.dec4 = nn.Linear(in_features=300, out_features=n_features)

    def encode(self, x):
        x = F.elu(self.enc1(x))
        x = F.elu(self.enc2(x))
        x = F.elu(self.enc3(x))
        return x

    def decode(self, x):
        x = F.elu(self.dec2(x))
        x = F.elu(self.dec3(x))
        x = self.dec4(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Autoencoder()
net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

def train(net, trainloader, NUM_EPOCHS):
    train_loss = []
    names = []
    images = []
    outputimages = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        
        for i, batch in enumerate(trainloader):
            '''n_samples/batch_size batches'''
            img, ncodes = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad() #sets the gradients to zero
            outputs = net(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i == 0 and epoch == 0:
                names.append(target_names[ncodes.tolist()])
                images.append([img.detach().cpu().numpy()])
                outputimages.append([outputs.detach().cpu().numpy()]) 
            
            if i == 0 and epoch == 99:
                names.append(target_names[ncodes.tolist()])
                images.append([img.detach().cpu().numpy()])
                outputimages.append([outputs.detach().cpu().numpy()]) 
                
            if i == 0 and epoch == 999:
                names.append(target_names[ncodes.tolist()])
                images.append([img.detach().cpu().numpy()])
                outputimages.append([outputs.detach().cpu().numpy()]) 
                               
                               
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
       
        if epoch % 100  == 0: #to plot the image reconstructions of the last batch of every 10 epochs 
            print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss))
    
        
    return train_loss, names, images, outputimages

train_loss, names, images, outputimages = train(net, trainloader, NUM_EPOCHS)


#plotting the original images and reconstructions
for i in range(0,len(names)):
    plot_portraits(outputimages[i][0], names[i], h, w, 3, 3)
    
#plot_portraits(X_test, test_names, h, w, 3, 3)
    
#plotting the loss against epoch number
plt.figure()
plt.plot(train_loss, 'k-')
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

encoded_train_images = np.zeros((len(train_names), 200))
encoded_test_images = np.zeros((len(test_names), 200))
for i,image in enumerate(train_data):
  temp_enc = net.encode(image[0].to(device))
  encoded_train_images[i,:] = temp_enc.detach().cpu().numpy()
for i,image in enumerate(test_data):
  temp_enc = net.encode(image[0].to(device))
  encoded_test_images[i,:] = temp_enc.detach().cpu().numpy()
  
# Train a SVC on reduced data
print("Fitting the classifier to the training set...")
t0 = time.time()
param_grid = {'n_neighbors': [1,2,3,4,5,6,7,8], }
clf = GridSearchCV(
    KNeighborsClassifier(weights='distance'), param_grid
)
clf = clf.fit(encoded_train_images, y_train)
print("done in %0.3fs" % (time.time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Quantitative evaluation of the model quality on the test set
print("Predicting people's names on the test set")
t0 = time.time()
y_pred = clf.predict(encoded_test_images)
print("done in %0.3fs" % (time.time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


    
    
    
