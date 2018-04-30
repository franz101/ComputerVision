#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:22:55 2018

@author: macbook
"""

import numpy as np
from sklearn.metrics import confusion_matrix
import glob
import pandas as pd
from imageio import imread


#1.1 Mittelwert der Kanäle
train_files = glob.glob('./haribo1/hariboTrain/*.png')
train_labels = [label.split("/")[-1].split("_")[0] for label in train_files]
train = [imread(img) for img in train_files]
test_files = glob.glob('./haribo1/hariboVal/*.png')
test_labels = [label.split("/")[-1].split("_")[0] for label in test_files]
test = [imread(img) for img in test_files]
#1.2 Mittelwert der Kanäle
def mittelwert_pro_kanal(imgs):
    result = []
    for img in imgs:
        r = np.mean(img[:,:,0])
        g = np.mean(img[:,:,1])
        b = np.mean(img[:,:,2])
        result.append([r,g,b])
    return np.array(result)

train_kanal_mittelwert = mittelwert_pro_kanal(train)
test_kanal_mittelwert  = mittelwert_pro_kanal(test)

distanzen = []
for test_mittelwert in test_kanal_mittelwert:
    test_ergebnis = []
    for train_mittelwert in train_kanal_mittelwert:
        distanz = np.linalg.norm(test_mittelwert-train_mittelwert)
        test_ergebnis.append(distanz)
    distanzen.append(test_ergebnis)
    
indices = np.argmin(np.array(distanzen),1) 
prediction_labels = np.take(train_labels, indices)
accuracy = np.mean(prediction_labels == test_labels)
# Akurazität von 16,66%.... Sehr schlecht.
#1.3

def histogram_pro_kanal(imgs):
    result = []
    for img in imgs:
        r = np.histogram(img[:,:,0], bins = 8, range = (0,256))[0]
        g = np.histogram(img[:,:,1], bins = 8, range = (0,256))[0]
        b = np.histogram(img[:,:,2], bins = 8, range = (0,256))[0]
        result.append(np.array([r,g,b]).flatten())
    return np.array(result)

train_histogram = histogram_pro_kanal(train)
test_histogram  = histogram_pro_kanal(test)

distanzen = []
for test_histo in test_histogram:
    test_ergebnis = []
    for train_histo in train_histogram:
        distanz = np.linalg.norm(test_histo-train_histo)
        test_ergebnis.append(distanz)
    distanzen.append(test_ergebnis)
    
indices = np.argmin(np.array(distanzen),1) 
prediction_labels = np.take(train_labels, indices)
accuracy = np.mean(prediction_labels == test_labels)
# Akurazität von 25%.... Sehr schlecht.

# Aufgabe 2:
from scipy.ndimage.morphology import binary_opening

test_binary = [binary_opening(img)[0] for img in test]
train_binary = [binary_opening(img)[0] for img in train]


"""from skimage import transform
>>> transform.rotate(img,20) #oder:
>>> from skimage.transform import rotate
>>> rotate(img,20)"""

### skimage.measure.regionprops


""" mask = mask.astype(np.int)
>>> props = regionprops(mask)[0]
>>> props.bbox#xMin,yMin,xMax,yMax"""
##mask = opening(mask, disk(5)).astype(np.uint8)