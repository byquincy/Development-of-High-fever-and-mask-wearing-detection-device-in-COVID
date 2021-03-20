#-*- coding:utf-8 -*-

import cv2
import numpy as np
import sys
from time import sleep
from pylepton import Lepton
import time
import os
import sys
import subprocess
#import threading

#----------변수 설정
face_eyes = cv2.CascadeClassifier('haarcascade_eye.xml')
#face_eyes = cv2.CascadeClassifier('haarcascade_righteye.xml')

face_mouth = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

cap = cv2.VideoCapture(0)

box_position_x = []
box_position_y = []

uptemp = 0
temp = 1
BMask = 0
BTemp = 0
TimePlaying = time.time()
NowPlaying = 0


#----------함수 설정
while True:
    BMask = 0
    BTemp = 0
    #Image Capture
    ret, img_frame = cap.read()
    print("Hmm1")
    with Lepton() as l:
        lepton_origin,_ = l.capture()
    print("Hmm2")
    
    #processing
    img_frame = cv2.resize(img_frame, dsize=(768, 576), interpolation=cv2.INTER_AREA)
    img_frame = img_frame[0:480, 128:768]
    
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    
    temp_img = cv2.resize(lepton_origin, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    cv2.normalize(temp_img, temp_img, 0, 65535, cv2.NORM_MINMAX) # extend contrast
    temp_img = cv2.convertScaleAbs(temp_img, alpha=(255.0/65535.0))
    
    temp_img = cv2.applyColorMap(temp_img, cv2.COLORMAP_JET)
    
    temp_img = cv2.rectangle(temp_img, (235,235), (245,245), (255, 255, 255), -1)
    
    
    #face detect
    start = time.time()
    faces = face_eyes.detectMultiScale(img_gray, 1.1, 4)
    box_position_x = []
    box_position_y = []
        
    for (x,y,w,h) in faces:
        if w < 30 and h < 30:
            temp = 1
            
            for i in range(0, len(box_position_x)):
                if box_position_x[i] - 100 < x and box_position_x[i] + 100 > x and box_position_y[i] - 100 < y and box_position_y[i] + 100 > y:
                    temp = 0
            
            if(x > 640 or y > 480):
                temp = 0
            elif(temp == 1):
                temp_data = float(lepton_origin[y/8][x/8])
                temp_data = temp_data - 8912
                temp_data = temp_data *7/4500
                temp_data = round(temp_data + 434/225, 2)
                
                if temp_data < 30 or temp_data > 40:
                    temp = 0
                    
                if uptemp == 1:
                    temp_data = 37.8
                    
            if temp == 1:
                cv2.putText(img_frame, str(temp_data), (x, y), 0, 1, (0, 0, 0))
                if temp_data > 37:
                    BTemp = 1
                    
                cv2.rectangle(img_frame,(x,y),(x+w,y+h),(0,255,0),2)
                box_position_x.append(x)
                box_position_y.append(y)
            else:
                cv2.rectangle(img_frame,(x,y),(x+w,y+h),(0,0,255),2)
                
    if(len(box_position_x) != 0):
        faces = face_mouth.detectMultiScale(img_gray, 1.03, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img_frame,(x,y),(x+w,y+h),(255,0,0),2)
            BMask = 1
    #print(time.time() - start)
    
    if BMask == 1 or BTemp == 1:
        print(time.time() - TimePlaying)
        if float(time.time() - TimePlaying) > float(NowPlaying):
            if BMask == 1 and BTemp == 1:
                subprocess.Popen(['aplay', 'Mask_Temp.wav'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
                NowPlaying = 3.5
                time.sleep(3.5)
                print("Mask, Temp")
            elif BMask == 1:
                subprocess.Popen(['aplay', 'Mask.wav'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
                NowPlaying = 1.5
                time.sleep(1.5)
                print("Mask")
            elif BTemp == 1:
                subprocess.Popen(['aplay', 'Temp.wav'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
                NowPlaying = 1.5
                time.sleep(1.5)
                print("Temp")
            TimePlaying = time.time()

    
    #result = cv2.add(temp_img, img_frame)
    result = img_frame
    #result = temp_img

    
    cv2.imshow('result', result)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 117:
        if uptemp == 1:
            uptemp = 0
        else:
            uptemp = 1
        print(uptemp)
cap.release()

