#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 08:53:15 2020

@author: niamhcray
"""
import matplotlib.pyplot as plt
import numpy as np
import os, time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA

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

#splits the full image set into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print('number of training images :', len(X_train))
print('number of test images :', len(X_test))


train_names = le.inverse_transform(y_train)
test_names = le.inverse_transform(y_test)

n_trainsamples = len(X_train)

n_components=200 #number of principal components

pca = PCA(n_components = n_components, svd_solver='arpack', whiten=True)
pca.fit(X_train) #fits the data to the PCA model
C = pca.components_ 
X_train_pca = pca.transform(X_train) #training set projected into facespace
X_test_pca = pca.transform(X_test)

#visualises the original images
plot_portraits(X_train, train_names, h, w, 3, 3)

#visualises the eigenfaces (of the training data)
eigenfaces = C.reshape((n_components, h, w))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_portraits(eigenfaces, eigenface_titles, h, w, 3, 3)

#visualises the reconstructed training images
encoded = np.matmul(C, X_train.T) 
reconstruction = (np.matmul(C.T, encoded)).T
reconstructed_faces = reconstruction.reshape(n_trainsamples, h, w)
plot_portraits(reconstructed_faces, train_names, h, w, 3, 3)

#K Nearest Neighbours Classifier
print("Fitting the classifier to the training set...")
t0 = time.time()
param_grid = {'n_neighbors': [1,2,3,4,5,6,7,8], }
clf = GridSearchCV(
    KNeighborsClassifier(weights='distance'), param_grid
)
clf = clf.fit(X_train_pca, y_train) 
print("done in %0.3fs" % (time.time() - t0))

print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Quantitative evaluation of the model quality on the test set
print("Predicting people's names on the test set...")
t0 = time.time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time.time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


