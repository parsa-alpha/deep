import cv2
import numpy as np
import os
import random


def get_data_mat(dir):
    X = []
    y = []
    
    files = os.listdir(dir)
    
    random.shuffle(files)
    for file in files:
        if file[-3:] == 'jpg':
            img = cv2.imread(os.path.join(dir, file))
            
            img = cv2.resize(img, (250, 150))
            
            img = img / 255.
            X.append(img)
            if file[:3] == 'air':
                y.append(0)
            elif file[:3] == 'fer':
                y.append(1)
            elif file[:3] == 'min':
                y.append(2)
            elif file[:3] == 'bea':
                y.append(3)
            elif file[:3] == 'pan':
                y.append(4)
        else:
            X, y = get_data_mat(os.path.join(dir, file))
            
    return np.array(X), np.array(y)
