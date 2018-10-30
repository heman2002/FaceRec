"""import cv2
import numpy as np
import os

casc_path = '/home/himanshu/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(casc_path)
video_capture = cv2.VideoCapture(0)

cnt=0
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print(faces)
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = frame[y:y+h, x:x+w]
        #roi = roi[:,:,::-1] #convert BGR to RGB
        roi = cv2.resize(roi, (200,200))
        cnt+=1
    #cv2.imshow('Video', frame)
    if cnt>0:
        break
cv2.imwrite('/home/himanshu/Pictures/Webcam/0.jpg', roi)        
#print(cnt)        
video_capture.release()
cv2.destroyAllWindows()"""
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import math
import requests
import json 
from PIL import Image

predictor_path = "/home/himanshu/Downloads/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
vs = VideoStream().start()
time.sleep(2.0)
cnt=0
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=1050)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    print(rects)	
    for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
	x = [shape[30][0], shape[8][0], shape[36][0], shape[45][0], shape[48][0], shape[54][0]]
	y = [shape[30][1], shape[8][1], shape[36][1], shape[45][1], shape[48][1], shape[54][1]]
	zipped = zip(x, y)
	img = frame
	size = img.shape
	#print(size)
	#vect[30], vect[8], vect[36], vect[45], vect[48], vect[54]    
	#2D image points. If you change the image, you need to change vector
	image_points = np.array(zipped, dtype="double")
 
	# 3D model points.
	model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])
 
 
	# Camera internals
 
	focal_length = size[1]
	#print(focal_length)
	center = (size[1]/2, size[0]/2)
	camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
 
	#print "Camera Matrix :\n {0}".format(camera_matrix)
 
	dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
 
	#print "Rotation Vector:\n {0}".format(rotation_vector)
	#print "Translation Vector:\n {0}".format(translation_vector)
 
 
	# Project a 3D point (0, 0, 1000.0) onto the image plane.
	# We use this to draw a line sticking out of the nose
 
 
	(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
 

	p1 = ( int(image_points[0][0]), int(image_points[0][1]))
	p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
	length = math.sqrt((int(image_points[0][0]) - int(nose_end_point2D[0][0][0]))**2 + (int(image_points[0][1]) - int(nose_end_point2D[0][0][1]))**2)
	left = rect.left()
	top = rect.top()
	right = rect.right()
	bottom = rect.bottom()
	if not length<40:
    		cnt=cnt-1
	 
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
        #for (x, y) in shape:
        #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	  
	# show the frame
   	cnt+=1
    #cv2.imshow("Frame", frame)
    #key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if cnt>0:
        break

roi = cv2.resize(img[top:bottom,left:right], (200, 200))
        
cv2.imwrite('/home/himanshu/Pictures/Webcam/0.jpg', roi) 
cv2.destroyAllWindows()
vs.stop()

files = {'image': open('/home/himanshu/Pictures/Webcam/0.jpg', 'rb')}
r = requests.post('http://localhost:8090/predict', files=files)
	#print(r)
#json_load = json.loads(r.text)
print(r.text)       
