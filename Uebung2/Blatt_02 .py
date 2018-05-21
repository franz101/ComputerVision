# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:54:31 2018

@author: 6Zhilins
"""
import numpy as np
import pandas as pd
import time

# Aufgabe 1
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
lenna = misc.imread('./Lenna.png')
lenna_rot = lenna[:,:,0]
fig, ax = plt.subplots(3, 4,figsize = (10,8))
#fig()
ax[0,0].imshow(lenna) #normal plotten
ax[0,0].set_title('Lenna') #einen sinnvollen Titel setzen
# 3.
ax[0,1].imshow(lenna_rot, cmap='Greys_r') #normal plotten
ax[0,1].set_title('Lenna Rot') #einen sinnvollen Titel setzen
ax[0,2].imshow(lenna_blau, cmap='Greys_r') #normal plotten
ax[0,2].set_title('Lenna Blau') #einen sinnvollen Titel setzen
ax[0,3].imshow(lenna_gruen, cmap='Greys_r') #normal plotten
ax[0,3].set_title('Lenna Gruen') #einen sinnvollen Titel setzen
#2.
lenna_blau = lenna[:,:,1]
lenna_gruen = lenna[:,:,2]
lenna_rgb = lenna_rot/3 + lenna_blau/3 + lenna_gruen/3

    
lenna_mean = np.sum(lenna_rgb,axis=(0,1))/3

ax[1,0].imshow(lenna_rgb, cmap='Greys_r') #normal plotten
ax[1,0].set_title('Lenna Mittelwert') #einen sinnvollen Titel setzen
# 4. 
lenna_shuffle = lenna.copy()
lenna_shuffle[:,:,0] = lenna[:,:,1]
lenna_shuffle[:,:,1] =lenna[:,:,2]
lenna_shuffle[:,:,2] = lenna[:,:,0]
# TODO: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.shuffle.html
ax[1,1].imshow(lenna_shuffle) #normal plotten
ax[1,1].set_title('Lenna Random') #einen sinnvollen Titel setzen

#5. Negativ
ax[1,2].imshow(255-lenna) #normal plotten
ax[1,2].set_title('Lenna Random')
print(np.mean(lenna))
print(np.std(lenna))