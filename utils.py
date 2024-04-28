import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.model_selection import train_test_split


# 讀入資料夾中所有的圖片，並將其存成np.array的形式
def read_data():
    f=[]
    m=[]
    count=0
    path1="Female_face"
    path2="Male_face"
    # load female pictures to numpy array
    items=os.listdir(path1)
    for i in items:
        item_path=os.path.join(path1,i)
        if os.path.isfile(item_path):
            data=cv2.imread(item_path)
            grey_image = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
            data=cv2.resize(grey_image,(128,128),interpolation=cv2.INTER_AREA)
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
            data=cv2.resize(grey_image,(128,128),interpolation=cv2.INTER_AREA)
            m.append(data)
        count+=1
        if count==1000:
            break
    m=np.array(m)
    print(m.shape)
    np.save("male.npy",m)


# 讀入np.array型別的資料，並切成training set 和 test set
def load_data():
   
    female=np.load("female.npy") 
    male=np.load("male.npy")
    data=np.concatenate((female,male),axis=0)
    
    male_label=np.ones(male.shape[0],dtype=int)
    female_label=np.zeros(female.shape[0],dtype=int)
    labels=np.concatenate((female_label,male_label),axis=0)
    
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


# 秀出預測後的結果
def show_prediction(images,prediction): 
  
    # 一張一張畫
    for i in range(0,images.shape[0]):

        idx=i
        idx%=10
        idx+=1
        # 建立子圖形2*5
        ax=plt.subplot(2,5,idx)

        # 畫出子圖形
        ax.imshow(images[i],cmap="gray")

        title="predict="+str(prediction[i])

        # 設定子圖形的標題大小
        ax.set_title(title,fontsize=10)

        # 設定不顯示刻度
        ax.set_xticks([]);ax.set_yticks([])
        if idx==10 or i==images.shape[0]-1:    
            plt.show()
        
