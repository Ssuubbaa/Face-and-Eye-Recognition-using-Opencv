#!/usr/bin/env python
# coding: utf-8

# In[38]:


#Face and Eye detection using HAAR Cascade Classifier


# # Face Detection

# In[ ]:


#import libraries
import cv2
import numpy as np


# In[66]:


face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# In[67]:


#reading and converting the image into grayscale
image = cv2.imread('modi.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


# In[68]:


faces = face_classifier.detectMultiScale(gray,1.3,5)


# In[69]:


#After the face detection, image will appear until the enter key pressed
if faces is  ():
    print("No faces found")
    
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('Face Detection',image)
    cv2.waitKey()
cv2.destroyAllWindows()


# # face and Eye Detection

# In[90]:


face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")


# In[91]:


image = cv2.imread('modi.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


# In[92]:


if faces is  ():
    print("No face is found")
    
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('face and Eye Detection',image)
    cv2.waitKey()
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(127,0,255),2)
        cv2.imshow('face and Eye Detection',image)
        cv2.waitKey()
cv2.destroyAllWindows()

