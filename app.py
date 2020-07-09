# -*- coding: utf-8 -*-
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2


# Define a flask app
app = Flask(__name__)



face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#

print('Model loaded. Check http://127.0.0.1:5000/')

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    face_haar_cascade=cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')#Load haar classifier
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles

    return faces,gray_img

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)
    
#Below function writes name of person for detected label
def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),4)
    
predicted_name = 'None'
def model_predict(img_path):
 #   img = image.load_img(img_path, target_size=(224, 224))
    test_img=cv2.imread(img_path)
    faces_detected,gray_img=faceDetection(test_img)

    name={0:"Reshwanth",1:"Venu" ,2:"Uma",3:"Upender",4:"Nani",5:"Naga",6:"Pavan",7:"Sada",8:"Nithish",9:"Uma T",10:"Prasad",11:"Nishanth",12:"Gayathri",13:"Sravanthi",14:"Prasanna",15:"Akshitha", 16:"Nikhila",17:"K.S.Chary"}#creating dictionary containing names for each label

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        draw_rect(test_img,face)
        predicted_name=name[label]
        #if confidence < 30:#If confidence less than 37 then don't print predicted face text on screen
        #       fr.put_text(test_img,predicted_name,x,y)
        if(confidence>37):#If confidence more than 37 then don't print predicted face text on screen
               continue
        put_text(test_img,predicted_name,x,y)

    resized_img=cv2.resize(test_img,(1000,1000))
    
    #cv2.imwrite(str(img_path) + 'waka.jpg',img)
    path = os.path.split(img_path)
    path = os.path.split(path[0])
    cv2.imwrite(os.path.join(path[0] ,'static','output', 'waka.jpg'), resized_img)
    new_path = os.path.join(path[0],'static' ,'output', 'waka.jpg')
    return new_path
    #cv2.waitKey(0)
    


@app.route('/', methods=['GET'])



def index():
    # Main page
    
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path)
                               
        return render_template('index.html', img_path=result)
        #return result
        
    return None

if __name__ == '__main__':
    app.run(debug=True)
