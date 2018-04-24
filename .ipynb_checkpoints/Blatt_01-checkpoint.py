# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:54:31 2018

@author: 6Zhilins
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import time

#1.1. Laden der Daten:
d = np.load('./trainingsDaten.npz')
tr_imgs = d['data']
tr_labels = d['labels']

t = np.load('./validierungsDaten.npz')
test_imgs = t['data']
test_labels = t['labels']


#Erstes Bild und Label auslesen:
img1 = tr_imgs[0,:,:] 
label1 = trLabels[0] 

#1.2. Mittelwert jedes Bildes
tock = time.time() #Die Ziet ist gestartet

median_of_training_data = np.median(tr_imgs,axis=(2,1))
median_of_test_data = np.median(test_imgs,axis=(2,1))

#1.3.
def euclidian_distance(a,b):
    return np.linalg.norm(a-b)

#1.4.
def predict_labels(measure_of_training_data, measure_of_test_data, tr_labels,test_labels):
    results = []
    for train_img in measure_of_training_data:
        results.append([euclidian_distance(train_img,b) for b in measure_of_test_data])

    minima = map(np.argmin, results)
    predicted_labels = [test_labels[x] for x in minima]
    return predicted_labels
    
def compare_train_test(predicted_labels, tr_labels,test_labels):
    evaluation = []
    for i,tr_label in enumerate(tr_labels):
        evaluation.append(tr_label == predicted_labels[i])
    return evaluation
    
prediction = predict_labels(median_of_training_data,median_of_test_data,  tr_labels,test_labels)
evaluation = compare_train_test(prediction, tr_labels,test_labels)
print("Accuracy: Total Images "+str(len(tr_labels))+". Correct predicted:"+\
str(np.sum(evalution)) + " Percentage: " + str((float(np.sum(evalution))/len(tr_labels))*100) +"%")
tick = time.time() #Timer ist gestoppt
tick_tock=tick-tock #bereche ob wie viele Zeit hat das geläuft
print()

# Ab 75% würde es zuverlässig sein? Statistisch akkurat relevant wäre es ab 95%
#2.
hist_of_training_data = [np.histogram(img, bins = 8, range = (0,256))[0] for img in tr_imgs]
hist_of_test_data = [np.histogram(img, bins = 8, range = (0,256))[0] for img in test_imgs]
prediction2 = predict_labels(hist_of_training_data,hist_of_test_data,  tr_labels,test_labels)

evaluation2 = compare_train_test(prediction2,  tr_labels,test_labels)
print("Accuracy: Total Images "+str(len(tr_labels))+". Correct predicted:"+\
str(np.sum(evaluation2)) + " Percentage: " + str((float(np.sum(evaluation2))/len(tr_labels))*100) +"%")
#3.
tock = time.time() #Timer on
con_matrix = confusion_matrix(tr_labels, prediction2)
con_matrix_df = pd.DataFrame(con_matrix, columns=[['Autos',"Hirsch","Schiff"]])
con_matrix_df.index = ['Autos',"Hirsch","Schiff"]
print(con_matrix_df)

tick = time.time() #Timer off
tick_tock2=tick-tock #berechen ob wie viele Zeit hat das geläuft

#bereche ob wann schnelle war
if tick_tock<tick_tock2:
    print("Erstes Mal war schnelle für " + str(tick_tock2-tick_tock) + " sec")
else:
    print("Zweites Mal war schnelle für " + str(tick_tock-tick_tock2) + " sec")