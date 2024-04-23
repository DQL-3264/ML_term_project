import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.model_selection import train_test_split

def data_preprocess():
    f=[]
    m=[]
    count=0
    path1="dataset/Female"
    path2="dataset/Male"
    # load female pictures to numpy array
    items=os.listdir(path1)
    for i in items:
        item_path=os.path.join(path1,i)
        if os.path.isfile(item_path):
            data=cv2.imread(item_path)
            grey_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data=cv2.resize(grey_image,(256,256),interpolation=cv2.INTER_AREA)
            f.append(data)
        count+=1
        if count==1000:
            break
    f=np.array(f)
    print(f.shape)
    np.save("female.npy",f)
    
    # load male pictures to numpy array
    count=0
    items=os.listdir(path2)
    for i in items:
        item_path=os.path.join(path2,i)
        if os.path.isfile(item_path):
            data=cv2.imread(item_path)
            grey_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data=cv2.resize(grey_image,(256,256),interpolation=cv2.INTER_AREA)
            m.append(data)
        count+=1
        if count==1000:
            break
    m=np.array(m)
    print(m.shape)
    np.save("male.npy",m)

def load_data():
   
    female=np.load("female.npy") 
    male=np.load("male.npy")
    data=np.concatenate((female,male),axis=0)
    
    male_label=np.ones(male.shape[0],dtype=int)
    female_label=np.zeros(female.shape[0],dtype=int)
    labels=np.concatenate((male_label,female_label),axis=0)
    
    X, x_test, Y, y_test = train_test_split(data, labels, test_size=400, random_state=42)
   
    return X,Y,x_test,y_test
    

def show_train_history(history):
    # Accuracy
    x_loc=MultipleLocator(2)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # Loss 
    x_loc=MultipleLocator(2)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Train History')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

