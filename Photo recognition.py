#-*- coding:utf-8 -*-

import cv2
import numpy as np
import sys
from time import sleep
import time
import os
import sys
import subprocess
import readchar

#----------변수 설정
face_eyes = cv2.CascadeClassifier('haarcascade_eye.xml')

face_mouth = cv2.CascadeClassifier('haarcascade_nose.xml')

all_face = 0
all_unmask = 0

enable_face = 0
enable_unmask = 0

total_sec = 0

"""print('input')
key = readchar.readchar()

if key == 'c':
    cap = cv2.VideoCapture(0)
    ret, img_frame = cap.read()
else:
    name = 'Test_Image/' + key + '.jpg'
    print(name)
    img_frame = cv2.imread(name)
img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)"""


#----------설정
BMask = 0

#face detect
for r in range(1, 61):
    name = str(r) + '.jpg'
    img_frame = cv2.imread('Test_Image/' + name)
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    
    start = time.time()
    faces = face_eyes.detectMultiScale(img_gray, 1.1, 4)
    if len(faces) != 0:
        faces = faces[faces[:,1].argsort()]
        faces = faces[faces[:,0].argsort()]
        face_group = np.empty((0, 2), int)
        
        i = 0
        BMask = 0
        for (x,y,w,h) in faces:
            out = 0
        
            if i != 0:
                #print('--' + str(i))
                for (searchX, searchY, searchW, searchH) in faces[0:i]:
                    #print(str(searchX) + ' <= ' + str(x) + ' <= ' + str(searchX + 2*searchW))
                    if searchX <= x and x <= searchX + 2*searchW:
                        #print('>>>')
                        #print(str(searchY - int(0.5*searchH)) + ' <= ' + str(y) + ' <= ' + str(searchY + int(1.5*searchH)))
                        if searchY - int(0.5*searchH) <= y and y <= searchY + int(1.5*searchH):
                            #print("Y")
                            out = 1
                            break
    
            if out == 1:
                cv2.rectangle(img_frame,(x,y),(x+w,y+h),(0,0,255),1)
                out = 0
            elif w < 50 and h < 50:
                cv2.rectangle(img_frame,(x,y),(x+w,y+h),(0,255,0),2)
                #cv2.putText(img_frame, '36.43', (x, y), 0, 1, (0, 255, 0))
                cv2.rectangle(img_frame,(x, y - int(0.5*h)),(x + 3*w, y + 3*h),(255,255,0),2)
            
                face_group = np.append(face_group, np.array([[i, None]]), axis=0)
            else:
                cv2.rectangle(img_frame,(x,y),(x+w,y+h),(0,0,255),1)
            i+=1
    
    
    if(len(face_group) != 0):
        unmask = face_mouth.detectMultiScale(img_gray, 1.03, 5)
        if len(unmask) != 0:
            unmask = unmask[unmask[:,0].argsort()]
            unmask = unmask[unmask[:,1].argsort()]
            
            i = 0
            for (x,y,w,h) in unmask:
                out = 1
                j = 0
                for (face_num, unmask_num) in face_group:
                    if unmask_num == None:
                        #print(str(searchX - int(0.3*searchW)) + ' <= ' + str(x) + ' <= ' + str(searchX + 2*searchW))
                        if (faces[face_num][0] - int(0.3*faces[face_num][2]) <= x and x <= faces[face_num][0] + 3*faces[face_num][2]) and \
                        (faces[face_num][0] - int(0.3*faces[face_num][2]) <= x+w and x+w <= faces[face_num][0] + 3*faces[face_num][2]):
                            #print(str(searchY) + ' <= ' + str(y) + ' <= ' + str(searchY + 2*searchH))
                            if faces[face_num][1] <= y and y <= faces[face_num][1] + 3*faces[face_num][3]:
                                out = 0
                                face_group[j][1] = i
                                break
                    j+=1
                if out == 1:
                    out = 0
                    cv2.rectangle(img_frame,(x,y),(x+w,y+h),(0,255,255,1))
                else:
                    cv2.rectangle(img_frame,(x,y),(x+w,y+h),(255,0,255),2)
                    BMask = 1
                    #cv2.putText(img_frame, 'No Mask', (x, y), 0, 1, (255, 0, 255))
                
                i+=1
                
    """if BMask == 1:
        print("Mask")"""
    
    end_time = time.time()
    
    result = img_frame
    
    line = name
    print(line)
    
    line = ' '
    line = line + str(len(faces)) + '\t/\t' + str(len(unmask)) + '\t||\t'
    line = line + str(len(face_group)) + '\t/\t' + str(len(np.where(None == face_group)[0])) + '\t||\t'
    line = line + str(end_time - start) + '\n'
    print(line)
    
    all_face += len(faces)
    all_unmask += len(unmask)
    enable_face += len(face_group)
    enable_unmask += len(np.where(None == face_group)[0])
    
    total_sec += float(end_time - start)
    
    name = 'Log/output' + name
    cv2.imwrite(name, result)
    
"""key = cv2.waitKey(0)

if key == 115:
    name = 'Log/output' + name
    cv2.imwrite(name, result)
    #print('image writing complete')"""

print('\n\nResult')
line = ' '
line = line + str(all_face) + '\t/\t' + str(all_unmask) + '\t||\t'
line = line + str(enable_face) + '\t/\t' + str(enable_unmask) + '\t||\t'
line = line + str(total_sec) + '\n'
print(line)