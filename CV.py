import cv2
import os
import numpy as np
from PIL import Image
# for i in os.listdir('./Female'):
#     print(i)
#     image = Image.open('./Female/'+str(i))
#     # print(image)
#     cropped = image.resize((256, 256))
#     cropped.save('./resize_Female/'+str(i))
count = 1
# img = cv2.imread('./Female/291.jpg')
# print(img.shape)
for i in os.listdir('./Female'):
    # print(str(i))
    img = cv2.imread('./Female/'+str(i))
    # print(img)
    if img is None:
        continue
    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_img = cv2.fastNlMeansDenoising(gray_img, h = 20, templateWindowSize=7)
    # OpenCV 裡的 CascadeClassifier() 方法 ( 級聯分類器 )，可以根據所提供的模型檔案，判斷某個事件是否屬於某種結果 
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")   # 載入人臉模型
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 3)    # 偵測人臉

    if(len(faces)!=0):
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)    # 利用 for 迴圈，抓取每個人臉屬性，繪製方框
            crop_img = img[y:y+h, x:x+w]

            cv2.imshow('oxxostudio', img)
            #cv2.imwrite('./Male_face/'+str(count)+'.jpg', crop_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # else:
    #     cv2.imwrite('./Female_face/'+str(count)+'.jpg', img)
    count = count+1